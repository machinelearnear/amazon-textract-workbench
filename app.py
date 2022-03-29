# Use Streamlit in Sagemaker Studio Lab
# Author: https://github.com/machinelearnear

# import dependencies
import pandas as pd
import streamlit as st
import boto3
import io
import json
from pathlib import Path
from PIL import Image

from textractgeofinder.ocrdb import AreaSelection
from textractgeofinder.tgeofinder import KeyValue, TGeoFinder, AreaSelection, SelectionElement
from textractcaller.t_call import Textract_Features, Textract_Types, call_textract
from textractprettyprinter.t_pretty_print import Pretty_Print_Table_Format, Textract_Pretty_Print, get_string, get_forms_string
from textractoverlayer.t_overlay import DocumentDimensions, get_bounding_boxes

import trp.trp2 as t2

# helper funcs
# -----------------------------------------------------------
def set_hierarchy_kv(list_kv: list[KeyValue], t_document: t2.TDocument, page_block: t2.TBlock, prefix="BORROWER"):
    """
    function to add "virtual" keys which we use to indicate context
    """
    for x in list_kv:
        t_document.add_virtual_key_for_existing_key(key_name=f"{prefix}_{x.key.text}",
                                                    existing_key=t_document.get_block_by_id(x.key.id),
                                                    page_block=page_block)

def add_sel_elements(t_document: t2.TDocument, selection_values: list[SelectionElement], key_base_name: str,
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
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def start_textract_client(credentials):
    return boto3.client(
        'textract',
        aws_access_key_id=credentials['Access key ID'].values[0],
        aws_secret_access_key=credentials['Secret access key'].values[0],
        region_name='us-east-2',
    )

def cached_call_textract(input_image, textract, options):
    return call_textract(
        input_document=image_to_byte_array(input_image),
        boto3_textract_client=textract, 
        features=options)

# streamlit app
# -----------------------------------------------------------
st.set_page_config(page_title='Textract Workbench', page_icon=":open_book:", layout="centered", initial_sidebar_state="auto", menu_items=None)

def main():
    st.title('Amazon Textract Workbench')
    with st.sidebar:
        st.header('Introduction')
        st.markdown('''
        This is a repo showing a quick start to taking advantage of the geometric context 
        found in a document to make tagging easier and more accurate with Amazon Textract. 
        We are going to be using SageMaker StudioLab as our dev environment and the 
        [Textractor](https://github.com/aws-samples/amazon-textract-textractor) 
        Python library by Martin Schade.
        ''')
        st.header('What does this do?')
        st.markdown('''
        It takes the output from the `AnalyzeText` Forms API and, combined with the `XY` 
        coordinates from the key/values detected, it allows you to tag these pairs into 
        groups for convenience, e.g. all "Patient" KV pairs.
        ''')

        # connect AWS credentials
        st.header('Add your `AWS` credentials to SM Studio Lab')
        st.markdown('Nothing will be saved locally, all is done on the fly.')
        credentials = pd.DataFrame()
        uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])
        
        if uploaded_file:
            credentials = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))
            st.success('File was read successfully')

        if not credentials.empty:
            textract = start_textract_client(credentials)
        else:
            st.warning('AWS credentials are not loaded.')
            
        # author
        st.info('Author: Nico Metallo')
        

    # read input image
    st.subheader('Read input image')
    options = st.selectbox('Please choose any of the following options',
        (
            'Choose sample from library',
            'Download image from URL',
            'Upload your own image',
        )
    )

    input_image = None
    if options == 'Choose sample from library':
        image_files = list(sorted([x for x in Path('test_images').rglob('*.jpg')]))
        selected_file = st.selectbox(
            'Select an image file from the list', image_files
        )
        st.write(f'You have selected `{selected_file}`')
        input_image = Image.open(selected_file)
    elif options == 'Download image from URL':
        image_url = st.text_input('Image URL')
        try:
            r = requests.get(image_url)
            input_image = Image.open(io.BytesIO(r.content))
        except Exception:
            st.error('There was an error downloading the image. Please check the URL again.')
    elif options == 'Upload your own image':
        uploaded_file = st.file_uploader("Choose file to upload")
        if uploaded_file:
            input_image = Image.open(io.BytesIO(uploaded_file.decode()))
            st.success('Image was successfully uploaded')

    if input_image:
        with st.expander("See uploaded image"):
            st.image(input_image, use_column_width=True)
            st.info('Note: Larger images will take longer to process.')
    else:
        st.warning('There is no image loaded.')
    
    # Define your K/V pairs
    st.subheader("Select key-value pairs as custom tags (optional)")
    number = st.number_input('Select the number of tags to use', min_value=0, value=0)
    save_locally = st.checkbox('Save tags locally as json file', value=True)
    tags = []
    
    if number == 0:
        st.warning('No tags selected.')
    else:
        with st.form("form_kv_pairs"):    
            container = st.container()
            for i in range(int(number)):
                    cA, cB, cC = st.columns(3)
                    with container:
                        textA = cA.text_input(f'ID_{i}_START_STR','Sample text',key=f'{i}_start_str')
                        textB = cB.text_input(f'ID_{i}_END_STR',key=f'{i}_end_str')
                        textC = cC.text_input(f'ID_{i}_PREFIX',f'PREFIX_{i}',key=f'{i}_prefix')
                        tags.append(
                            {
                                f'ID_{i}_START_STR': textA,
                                f'ID_{i}_END_STR': textB,
                                f'ID_{i}_PREFIX': textC,
                            }
                        )

            # Every form must have a submit button.
            submitted = st.form_submit_button("Save")

        if submitted:
            st.success('Saved changes.')
            with st.expander("See list of selected tags"):
                st.write(tags)
        else:
            st.warning('No tags selected.')
    
    # Get OCR predictions
    st.subheader('Run Amazon Textract')
    st.write('')
    features = [str(v) for k,v in enumerate(Textract_Features)]
    options = st.multiselect(
        'Select additional actions to use from Amazon Textract',
        [v for k,v in enumerate(Textract_Features)],
        help='Source: https://docs.aws.amazon.com/textract/latest/dg/API_Operations.html')

    t_json = None
    t_df = pd.DataFrame()

    if input_image:
        if not credentials.empty:
            if st.button('Get OCR Predictions'):
                try:
                    t_json = cached_call_textract(input_image, textract, options)
                    
                    # convert json > string > csv > pandas
                    t_df = t_json_to_t_df(t_json)
                    
                    st.success('Done!')
                except Exception as e:
                    st.error(e)
                    
            if tags:
                if st.button('Get Custom KV Pairs'):
                    if t_json:
                        # we use this information to add new key value pairs to the Amazon Textract Response JSON schema
                        t_document = tag_kv_pairs_to_text(t_json, tags)
                        
                        # convert json > string > csv > pandas
                        t_df = t_json_to_t_df(t2.TDocumentSchema().dump(t_document))

                        st.success('Done!')
                    else:
                        st.warning('Please run "Get OCR predictions" first')
            else:
                st.warning('No tags selected')
                
            if t_json and not t_df.empty:
                st.subheader('See results')
                with st.expander('Expand to see image with overlayed predictions'):
                    document_dimension:DocumentDimensions = DocumentDimensions(
                        doc_width=input_image.size[0], doc_height=input_image.size[1])
                    overlay=[Textract_Types.WORD, Textract_Types.CELL]

                    # bounding_box_list = get_bounding_boxes(
                    #     textract_json=t_json,
                    #     document_dimensions=document_dimension,
                    #     overlay_features=overlay)
                    
                with st.expander('Amazon Textract Response as JSON'):
                    st.write(t_json)
                    st.download_button("Download json",json.dumps(t_json))
                    
                with st.expander('Key-value pairs as DataFrame'):
                    st.write(t_df)
                    
                    csv = convert_df(t_df)
                    st.download_button(
                         label="Download table as CSV",
                         data=csv,
                         file_name='textract_output.csv',
                         mime='text/csv',
                     )
        else:
            st.warning('AWS credentials are not loaded.')
    else:
        st.warning('There is no image loaded.')
        
    
    
    # footer
    st.header('References')
    st.markdown('''
    - [Extract Information By Using Document Geometry + Amazon Textract](https://github.com/machinelearnear/extract-info-by-doc-geometry-aws-textract)
    - [Access AWS resources from Studiolab](https://github.com/aws/studio-lab-examples/blob/main/connect-to-aws/Access_AWS_from_Studio_Lab.ipynb)
    - [Textractor GeoFinder Sample Notebook](https://github.com/aws-samples/amazon-textract-textractor/blob/master/tpipelinegeofinder/geofinder-sample-notebook.ipynb)
    - [Intelligent Document Processing Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/c2af04b2-54ab-4b3d-be73-c7dd39074b20/en-US/)
    ''')

# run application
# -----------------------------------------------------------
if __name__ == '__main__':
    main()