# Amazon Textract Workbench
# Author: https://github.com/machinelearnear

# import dependencies
# -----------------------------------------------------------
import pandas as pd
import streamlit as st
import boto3
import io
import json
import re
import os
os.makedirs("dependencies",exist_ok=True)
import sys
import cv2
import numpy as np 
import tensorflow as tf

from transformers import pipeline
from matplotlib import pyplot
from pathlib import Path
from PIL import Image
from typing import List, Optional
from tabulate import tabulate
from pdf2image import convert_from_path, convert_from_bytes
from streamlit_image_comparison import image_comparison

from textractgeofinder.ocrdb import AreaSelection
from textractgeofinder.tgeofinder import KeyValue, TGeoFinder, AreaSelection, SelectionElement
from textractcaller.t_call import Textract_Features, Textract_Types, call_textract
from textractprettyprinter.t_pretty_print import Pretty_Print_Table_Format, Textract_Pretty_Print, get_string, get_forms_string
from textractoverlayer.t_overlay import DocumentDimensions, get_bounding_boxes

import trp.trp2 as t2

# PRE-PROCESSING (1)
# -----------------------------------------------------------
# Source: https://github.com/Leedeng/SauvolaNet
# -----------------------------------------------------------
from os.path import exists as path_exists
path_repo_sauvolanet = 'dependencies/SauvolaNet'
if not path_exists(path_repo_sauvolanet):
    os.system(f'git clone https://github.com/Leedeng/SauvolaNet.git {path_repo_sauvolanet}')
sys.path.append(f'{path_repo_sauvolanet}/SauvolaDocBin/')
pd.set_option('display.float_format','{:.4f}'.format)
from dataUtils import collect_binarization_by_dataset, DataGenerator
from testUtils import prepare_inference, find_best_model
from layerUtils import *
from metrics import *

@st.cache
def sauvolanet_load_model(model_root = f'{path_repo_sauvolanet}/pretrained_models/'):
    for this in os.listdir(model_root) :
        if this.endswith('.h5') :
            model_filepath = os.path.join(model_root, this)
            model = prepare_inference(model_filepath)
            print(model_filepath)
    return model

def sauvolanet_read_decode_image(model,im):
    rgb = np.array(im)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x = gray.astype('float32')[None, ..., None]/255.
    pred = model.predict(x)
    return Image.fromarray(pred[0,...,0] > 0)

# PRE-PROCESSING (2)
# -----------------------------------------------------------
# Source: Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)
# by Kai Zhang (2021/05-2021/11)
# -----------------------------------------------------------

path_repo_SCUNet = 'dependencies/SCUNet'
if not path_exists(path_repo_SCUNet):
    os.system(f'git clone https://github.com/cszn/SCUNet.git {path_repo_SCUNet}')
    os.system(f'wget https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth -P {path_repo_SCUNet}/model_zoo')
from datetime import datetime
from collections import OrderedDict
import torch
sys.path.append(f'{path_repo_SCUNet}')
from utils import utils_model
from utils import utils_image as util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_channels = 3

@st.cache
def scunet_load_model(model_path=f'{path_repo_SCUNet}/model_zoo/scunet_color_real_psnr.pth'):
    from models.network_scunet import SCUNet as net
    model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
        
    return model.to(device)
    
def scunet_inference(model,img):
    # ------------------------------------
    # (1) img_L
    # ------------------------------------
    img_L = np.asarray(img)
    if img_L.ndim == 2:
        img_L = cv2.cvtColor(img_L, cv2.COLOR_GRAY2RGB)  # GGG
    else:
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)  # RGB
    
    img_L = util.uint2single(img_L)
    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)
    
    # ------------------------------------
    # (2) img_E
    # ------------------------------------
    img_E = model(img_L)
    img_E = util.tensor2uint(img_E)

    return Image.fromarray(img_E)

# helper funcs
# -----------------------------------------------------------
def set_hierarchy_kv(list_kv, t_document: t2.TDocument, page_block: t2.TBlock, prefix="BORROWER"):
    """
    function to add "virtual" keys which we use to indicate context
    """
    for x in list_kv:
        t_document.add_virtual_key_for_existing_key(key_name=f"{prefix}_{x.key.text}",
                                                    existing_key=t_document.get_block_by_id(x.key.id),
                                                    page_block=page_block)

def add_sel_elements(t_document: t2.TDocument, selection_values, key_base_name: str,
                     page_block: t2.TBlock) -> t2.TDocument:
    """
    Function that makes it easier to add selection elements to the Amazon Textract Response JSON schema
    """
    for sel_element in selection_values:
        sel_key_string = "_".join([s_key.original_text.upper() for s_key in sel_element.key if s_key.original_text])
        if sel_key_string:
            if sel_element.selection.original_text:
                t_document.add_virtual_key_for_existing_key(page_block=page_block,
                                                            key_name=f"{key_base_name}->{sel_key_string}",
                                                            existing_key=t_document.get_block_by_id(
                                                                sel_element.key[0].id))
    return t_document

def tag_kv_pairs_to_text(textract_json, list_of_items):
    t_document, geofinder_doc = geofinder(textract_json)
    for item in list_of_items:
        start = geofinder_doc.find_phrase_on_page(item['first_str'])[0]
        if len(item) == 3:
            end = geofinder_doc.find_phrase_on_page(item['last_str'], min_textdistance=0.99)[0]
            top_left = t2.TPoint(y=start.ymax, x=0)
            lower_right = t2.TPoint(y=end.ymin, x=doc_width)
            form_fields = geofinder_doc.get_form_fields_in_area(
                area_selection=AreaSelection(top_left=top_left, lower_right=lower_right, page_number=1))
            set_hierarchy_kv(
                list_kv=form_fields, t_document=t_document, prefix=item['prefix'], page_block=t_document.pages[0])
        else:
            top_left = t2.TPoint(y=start.ymin - 50, x=0)
            lower_right = t2.TPoint(y=start.ymax + 50, x=doc_width)
            sel_values: list[SelectionElement] = geofinder_doc.get_selection_values_in_area(area_selection=AreaSelection(
                top_left=top_left, lower_right=lower_right, page_number=1),
                                                                                            exclude_ids=[])
            t_document = add_sel_elements(t_document=t_document,
                             selection_values=sel_values,
                             key_base_name=item['prefix'],
                             page_block=t_document.pages[0])
            
    return t_document
    
def geofinder(textract_json, doc_height=1000, doc_width=1000):
    j = textract_json.copy()
    t_document = t2.TDocumentSchema().load(j)
    geofinder_doc = TGeoFinder(j, doc_height=doc_height, doc_width=doc_width)
    
    return t_document, geofinder_doc

def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    
    return imgByteArr

def t_json_to_t_df(t_json):
    # convert t_json > string > csv
    t_str = get_string(
        textract_json=t_json,
        table_format=Pretty_Print_Table_Format.csv,
        output_type=[Textract_Pretty_Print.FORMS],
    )

    return pd.read_csv(io.StringIO(t_str), sep=",")

@st.cache
def convert_pandas(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.experimental_singleton
def start_textract_client(credentials):
    return boto3.client(
        'textract',
        aws_access_key_id=credentials['Access key ID'].values[0],
        aws_secret_access_key=credentials['Secret access key'].values[0],
        region_name='us-east-2',
    )

def return_fnames(folder, extensions={'.png','.jpg','.jpeg'}):
    f = (p for p in Path(folder).glob("**/*") if p.suffix in extensions)
    return [x for x in f if 'ipynb_checkpoints' not in str(x)]

@st.cache
def return_anno_file(folder, image_fname):
    files = list(sorted([x for x in Path(folder).rglob('*.json')]))
    selected = [x for x in files if 
                Path(image_fname).stem in str(x) and 'ipynb_checkpoints' not in str(x)]
    return selected[0]

def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def add_item_to_input_queries_list(text, alias='', verbose=False):
    text = re.sub('[^\w\s]','', text)
    if len(alias)==0:
        alias = text.replace(' ','_').upper()

    if not any(text in x['Text'] for x in st.session_state.input_queries):
        st.session_state.input_queries.append(
            {
                "Text": text, 
                "Alias": alias
            }
        )
        if verbose: st.success('Added')
    else:
        if verbose: st.warning('Already exists')

def remove_item():
    del st.session_state.input_queries[st.session_state.index]
    
def clear_all_items():
    st.session_state.input_queries = []
    st.session_state.index = 0

def parse_response(response):
    from trp import Document
    doc = Document(response)
    text = ''
    for page in doc.pages:
        for line in page.lines:
            for word in line.words:
                text = text + word.text + ' '
    return text.strip()

@st.experimental_memo
def hf_pipeline(model_name, task):
    return pipeline(task, model=model_name)

# streamlit app
# -----------------------------------------------------------
st.set_page_config(
    page_title='Textract Workbench', 
    page_icon=":open_book:", 
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None)

def main():
    # intro and sidebar
    #######################
    st.title('Amazon Textract Workbench v0.1')
    st.markdown('''
    This web app shows you a step by step tutorial on how to take advantage of the geometric
    context found in an image to make the tagging of key and value pairs easier and more accurate
    with [Amazon Textract](https://aws.amazon.com/textract/). We are running this demo on top of
    [SageMaker Studio Lab](https://www.youtube.com/watch?v=FUEIwAsrMP4) using the 
    [Textractor](https://github.com/aws-samples/amazon-textract-textractor) library developed by 
    [Martin Schade](https://www.linkedin.com/in/martinschade/) et al.
    ''')
    
    with st.sidebar:
        # about
        st.subheader('About this demo')
        st.markdown('''
        Built by Nicolás Metallo (metallo@amazon.com)
        ''')
        st.markdown(f'This web app is running on `{device}`')
        
        # connect AWS credentials
        with st.expander('Connect your AWS credentials'):
            st.markdown('Required to use Amazon Textract. No data is stored locally, only streamed to memory.')
            credentials = pd.DataFrame()
            uploaded_file = st.file_uploader("Upload your csv file", type=['csv'], key='uploaded_file_credentials')

            if uploaded_file:
                credentials = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))

        if not credentials.empty:
            textract_client = start_textract_client(credentials)
            st.success('AWS credentials are loaded.')
        else:
            st.warning('AWS credentials are not loaded.')      

    # (1) read input image
    #######################
    st.header('(1) Read input image')
    options = st.selectbox('Please choose any of the following options',
        (
            'Choose sample image from library',
            'Download image from URL',
            'Upload your own image',
        )
    )

    input_image = None
    if options == 'Choose sample image from library':
        image_files = return_fnames('test_images')
        selected_file = st.selectbox(
            'Select an image file or PDF from the list', image_files
        )
        image_fname = selected_file
        st.write(f'You have selected `{image_fname}`')
        
        if Path(image_fname).suffix != '.pdf':
            input_image = Image.open(selected_file)
        else:
            input_image = convert_from_path(selected_file,fmt='png')[0] # only first page
            
    elif options == 'Download image from URL':
        image_url = st.text_input('Image URL')
        try:
            r = requests.get(image_url)
            image_fname = get_filename_from_cd(r.headers.get('content-disposition'))
            input_image = Image.open(io.BytesIO(r.content))
        except Exception:
            st.error('There was an error downloading the image. Please check the URL again.')
    elif options == 'Upload your own image':
        uploaded_file = st.file_uploader("Choose file to upload", key='uploaded_file_input_image')
        if uploaded_file:
            if Path(uploaded_file.name).suffix != '.pdf':
                input_image = Image.open(io.BytesIO(uploaded_file.decode()))
            else:
                input_image = convert_from_bytes(uploaded_file.read(),fmt='png')[0] # only first page
            st.success('Image was successfully uploaded')

    if input_image:
        max_im_size = (1000,1000)
        input_image.thumbnail(max_im_size, Image.Resampling.LANCZOS)
        with st.expander("See input image"):
            st.image(input_image, use_column_width=True)
            st.info(f'The input image has been resized to fit within `{max_im_size}`')
    else:
        st.warning('There is no image loaded.')
        
    # image pre-processing
    #######################
    st.subheader('(Optional) Image pre-processing')
    options_preproc = st.selectbox('Please choose any of the following options',
        (
            'No pre-processing',
            'SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization (ICDAR2021)',
            'Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis',
            'Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR2022)',
        )
    )
    
    modified_image = None
    if input_image:
        if options_preproc == 'SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization (ICDAR2021)':
            try:
                with st.spinner():
                    sauvolanet_model = sauvolanet_load_model()
                    modified_image = sauvolanet_read_decode_image(sauvolanet_model,input_image)
                    st.success('Done!')
            except Exception as e:
                st.error(e)
        if options_preproc == 'Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis':
            try:
                with st.spinner():
                    scunet_model = scunet_load_model()
                    modified_image = scunet_inference(scunet_model,input_image)
                    st.success('Done!')
            except Exception as e:
                st.error(e)
                
        if modified_image:
            with st.expander("See modified image"):
                image_comparison(
                    img1=input_image, img2=modified_image,
                    label1='Original', label2='Modified',
                )
    else:
        st.warning('There is no image loaded.')
    
    # (2) retrieve ocr preds
    #######################
    st.header('(2) Amazon Textract')
    if not 'response' in st.session_state:
        st.session_state.response = None
    feature_types=[]
    cAA, cBA = st.columns(2)
    with cAA:
        options = st.selectbox(
             'The following actions are supported:',
             ('DetectDocumentText','AnalyzeDocument','AnalyzeExpense','AnalyzeID'),
            help='Read more: https://docs.aws.amazon.com/textract/latest/dg/API_Operations.html')
        st.write(f'You selected: `{options}`')
    with cBA:
        if options == 'AnalyzeDocument':
            feature_types = st.multiselect(
                'Select feature types for "AnalyzeDocument"',
                ['TABLES','FORMS','QUERIES'],
                help='Read more: https://docs.aws.amazon.com/textract/latest/dg/API_AnalyzeDocument.html')
            st.write(f'You selected: `{feature_types}`')
        
    if 'QUERIES' in feature_types:
        if 'input_queries' not in st.session_state:
            st.session_state.input_queries = []
        if 'index' not in st.session_state:
            st.session_state.index = 0
        
        with st.expander('Would you like to upload your existing list of queries?'):
            uploaded_file = st.file_uploader("Choose file to upload", key='uploaded_file_queries')
            if uploaded_file:
                queries = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                queries = [x.strip().lower() for x in queries.split(',')]
                for x in queries: add_item_to_input_queries_list(x)
                st.success('List of queries was successfully uploaded')   

        with st.expander('Input your new queries here'):
            cAB, cBB = st.columns([3,1])
            with cAB:
                st.text_input('Input your new query',
                              key='add_query_text',
                              help='Input queries that Textract will use to extract the data that is most important to you.')
            with cBB:
                st.text_input('Alias (Optional)',
                              key='add_query_alias')

            if st.button('+ Add query') and len(st.session_state.add_query_text)>0:
                input_query_text = st.session_state.add_query_text.strip()
                input_query_alias = st.session_state.add_query_alias
                add_item_to_input_queries_list(input_query_text, input_query_alias, verbose=True)
            
        if len(st.session_state.input_queries)==0: 
            st.warning('No queries selected')
        else:
            with st.expander('Edit existing queries'):
                cAC, cBC = st.columns([3,1])
                cAC.write(st.session_state.input_queries)
                cBC.number_input(
                    'Select entry number',
                    min_value=0,
                    max_value=len(st.session_state.input_queries)-1,
                    key='index',
                )
                if cBC.button('Remove item',on_click=remove_item):
                    st.success('Deleted!')
                if cBC.button('Clear all',on_click=clear_all_items):
                    if len(st.session_state.input_queries)==0:
                        st.success('Cleared!')
                    else:
                        cBC.warning('Delete the uploaded file to clear all')
                    
    st.subheader('Run and review response')
    if input_image:
        if not credentials.empty:
            aa, bb = st.columns([1,5])
            if aa.button('✍ Submit'):
                st.session_state.response = None
                if options == 'AnalyzeDocument' and feature_types:
                    if 'QUERIES' in feature_types:
                        if len(st.session_state.input_queries)==0:
                            st.error('Please add queries before you submit your request with QUERIES')
                        else:
                            response = textract_client.analyze_document(
                                Document = {
                                    'Bytes': image_to_byte_array(input_image),
                                },
                                FeatureTypes = feature_types,
                                QueriesConfig={
                                    'Queries': st.session_state.input_queries[:15] # max queries per page: 15
                                }
                            )
                    else:
                        response = textract_client.analyze_document(
                            Document = {
                                'Bytes': image_to_byte_array(input_image),
                            },
                            FeatureTypes = feature_types,
                        )
                elif options == 'AnalyzeExpense':
                    response = textract_client.analyze_expense(
                        Document={
                            'Bytes': image_to_byte_array(input_image),
                        }
                    )
                elif options == 'AnalyzeID':
                    response = textract_client.analyze_id(
                        DocumentPages=[
                            {
                                'Bytes': image_to_byte_array(input_image),
                            },
                        ]
                    )
                else:
                    response = textract_client.detect_document_text(
                        Document={
                            'Bytes': image_to_byte_array(input_image),
                        }
                    )
                    
                if response:
                    aa.success('Finished!')
                    with bb.expander('View response'):
                        st.markdown('**RAW TEXT**')
                        output_text = parse_response(response)
                        st.write(output_text)
                        st.markdown('**JSON**')
                        st.write(response)
                        
                    if feature_types:
                        with bb.expander('View response from AnalyzeDocument'):
                            if 'QUERIES' in feature_types:
                                st.markdown('**QUERIES**')
                                d = t2.TDocumentSchema().load(response)
                                page = d.pages[0]
                                query_answers = d.get_query_answers(page=page)
                                queries_df = pd.DataFrame(
                                    query_answers,
                                    columns=['Text','Alias','Value'])
                                st.dataframe(queries_df)
                            if 'FORMS' in feature_types:
                                st.markdown('**FORMS**')
                                forms = get_string(
                                    textract_json=response,
                                    table_format=Pretty_Print_Table_Format.csv,
                                    output_type=[Textract_Pretty_Print.FORMS],
                                )
                                forms_df = pd.read_csv(io.StringIO(forms),sep=",")
                                st.dataframe(forms_df)
                            if 'TABLES' in feature_types:
                                st.markdown('**TABLES**')
                                tables = get_string(
                                    textract_json=response,
                                    table_format=Pretty_Print_Table_Format.csv,
                                    output_type=[Textract_Pretty_Print.TABLES],
                                )
                                tables_df = pd.read_csv(io.StringIO(tables),sep=",")
                                st.dataframe(tables_df)
                                
                    if options == 'AnalyzeExpense':
                        with st.expander('View response from AnalyzeExpense'):
                            pass
                        
                    if options == 'AnalyzeID':
                        with st.expander('View response from AnalyzeID'):
                            pass
                        
                st.session_state.response = response
            elif not st.session_state.response:
                st.warning('No response generated')                   
        else:
            st.warning('AWS credentials are not loaded.')
    else:
        st.warning('There is no image loaded.')
    
    # expand with Amazon Comprehend and Hugging Face
    #######################
    st.header('(3) Amazon Comprehend')
    st.header('(4) Hugging Face Transformers')
    st.subheader("Summary")
    if input_image:
        if st.session_state.response:
            options = st.selectbox(
                'Please select any of the following to review the Textract response',
                ['Not selected','google/pegasus-xsum','facebook/bart-large-cnn'],
                help='https://huggingface.co/models?pipeline_tag=summarization&sort=downloads',
            )
            st.write(f'You selected: `{options}`')
            
            if options != 'Not selected':
                with st.spinner('Downloading model weights and loading...'):
                    pipe = pipeline("summarization", model=options)
                summary = pipe(parse_response(st.session_state.response), 
                               max_length=130, min_length=30, do_sample=False)
                
                with st.expander('View response'):
                    st.write(summary)
        else:
            st.warning('No response generated')
    else:
        st.warning('There is no image loaded.')
    
    # footer
    st.header('References')
    st.code(
        '''
    @INPROCEEDINGS{9506664,  
      author={Li, Deng and Wu, Yue and Zhou, Yicong},  
      booktitle={The 16th International Conference on Document Analysis and Recognition (ICDAR)},   
      title={SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization},   
      year={2021},  
      volume={},  
      number={},  
      pages={538–553},  
      doi={https://doi.org/10.1007/978-3-030-86337-1_36}}
  
    @article{zhang2022practical,
    title={Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis},
    author={Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Timofte, Radu and Van Gool, Luc},
    journal={arXiv preprint},
    year={2022}
    }
        '''
        , language='bibtex')
    
    st.header('Disclaimer')
    st.markdown('''
    - The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
    - The ideas and opinions outlined in these examples are my own and do not represent the opinions of AWS.
    ''')

# run application
# -----------------------------------------------------------
if __name__ == '__main__':
    main()