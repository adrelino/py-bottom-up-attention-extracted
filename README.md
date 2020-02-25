# Bottom-up Attention using Detectron2

Extracted diff of https://github.com/airsplay/py-bottom-up-attention/ and detectron2.

**Previewing**
The detectron2 system with **exact same model and weight** as the Caffe VG Faster R-CNN provided in [bottom-up-attetion](https://github.com/peteanderson80/bottom-up-attention).

## Installation
```
git clone https://github.com/adrelino/py-bottom-up-attention-extracted.git
cd py-bottom-up-attention-extracted
# Install python libraries
pip install -r requirements.txt
```

Install [detectron2](https://github.com/facebookresearch/detectron2.git)

## Demos

### Object Detection
[demo vg detection](demo/demo_vg_detection.ipynb)

### Feature Extraction 
1. Single image: [demo extraction](demo/demo_feature_extraction.ipynb)
2. Batchwise extraction: [demo batchwise extraction](demo/demo_batchwise_feature_extraction.ipynb)

## Feature Extraction Scripts for LXMERT
1. For MS COCO (VQA): [vqa script](demo/detectron2_mscoco_proposal_maxnms.py)