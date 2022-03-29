# Use Streamlit in Sagemaker Studio Lab
# Author: https://github.com/machinelearnear

# import dependencies
import pandas as pd
import streamlit as st
import boto3
import io
from pathlib import Path
from PIL import Image

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
            textract = boto3.client(
                'textract',
                aws_access_key_id=credentials['Access key ID'].values[0],
                aws_secret_access_key=credentials['Secret access key'].values[0],
                region_name='us-east-2',
            )
        else:
            st.warning('There is no csv loaded')
            
        # author
        st.info('Author: Nico Metallo')
        

    # read input image
    st.subheader('Read input image')
    options = st.radio('Please choose any of the following options',
        (
            'Choose example from library',
            'Download image from URL',
            'Upload your own image',
        )
    )

    input_image = None
    if options == 'Choose example from library':
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
        st.image(input_image, use_column_width=True)
        st.info('Note: Larger images will take longer to process.')
    else:
        st.warning('There is no image loaded.')
    
    # Define your K/V pairs
    st.subheader("Select your own key-value pairs")
    number = st.number_input('Select the number of tags to use', min_value=1)
    save_locally = st.checkbox('Save tags locally as json file', value=True)
    tags = []
    
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
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.success('Saved changes.')
        with st.expander("See list of selected tags"):
            st.write(tags)
    else:
        st.warning('No tags selected.')
    
    # run Textract
    st.header('Run Amazon Textract')
    st.write('')
    if input_image and st.button('Submit'):
        try:
            pass
            # with st.spinner():
                # output = infer(model, np.asarray(input_image))
                # output_image = Image.fromarray((output * 255).astype(np.uint8))
                # image_comparison(
                #     img1=input_image, img2=output_image,
                #     label1='Original', label2='Depth Estimation',
                # )
        except Exception as e:
            st.error(e)
            st.error('There was an error processing the input image')
    if not input_image: st.warning('There is no image loaded')
    
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