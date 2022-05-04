# Amazon Textract Workbench
This repo shows you a step by step tutorial on how to take advantage of the geometric context detected in an image to make the tagging of key and value pairs easier and more accurate with [Amazon Textract](https://aws.amazon.com/textract/). We are running this demo on top of [SageMaker Studio Lab](https://www.youtube.com/watch?v=FUEIwAsrMP4) using the [Textractor](https://github.com/aws-samples/amazon-textract-textractor) library developed by [Martin Schade](https://www.linkedin.com/in/martinschade/) et al. We will also make use of the recently launched ["Queries"](https://aws.amazon.com/blogs/machine-learning/specify-and-extract-information-from-documents-using-the-new-queries-feature-in-amazon-textract/) functionality in Textract. 

The way this works is by taking the output from the `AnalyzeDocument` API and, combined with the XY coordinates from the key/values detected, tagging these pairs into convenient groups, e.g. all "Patient" KV pairs.


## Getting started
- [SageMaker StudioLab Explainer Video](https://www.youtube.com/watch?v=FUEIwAsrMP4)
- [How to extract information by using document geometry & Amazon Textract](https://github.com/machinelearnear/extract-info-by-doc-geometry-aws-textract)
- [How to access AWS resources from Studiolab](https://github.com/aws/studio-lab-examples/blob/main/connect-to-aws/Access_AWS_from_Studio_Lab.ipynb)
- [Amazon Textractor Python Library](https://github.com/aws-samples/amazon-textract-textractor)
- [Textractor GeoFinder Sample Notebook](https://github.com/aws-samples/amazon-textract-textractor/blob/master/tpipelinegeofinder/geofinder-sample-notebook.ipynb)
- [Intelligent Document Processing Workshop by AWS](https://catalog.us-east-1.prod.workshops.aws/workshops/c2af04b2-54ab-4b3d-be73-c7dd39074b20/en-US/)
- [Specify and extract information from documents using the new Queries feature in Amazon Textract](https://aws.amazon.com/blogs/machine-learning/specify-and-extract-information-from-documents-using-the-new-queries-feature-in-amazon-textract/)
- [Sample notebook showing use of new "Queries" feature](https://github.com/aws-samples/amazon-textract-code-samples/blob/master/python/queries/paystub.ipynb)

## Additional resources
- [Annotated Text Component for Streamlit](https://github.com/tvst/st-annotated-text)

## How to start
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/machinelearnear/amazon-textract-workbench/blob/main/launch_app.ipynb)

## References

```bibtex
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
```

## Disclaimer
- The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
- The ideas and opinions outlined in these examples are my own and do not represent the opinions of AWS.