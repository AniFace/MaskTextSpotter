# MaskTextSpotter
## Main problem that our project is targeting
Given multiple object detection algorithms implemented in detectron2, and the importance of OCR problems, we want to implement Mask TextSpotter algorithm aiming at OCR task under detectron2 framework to simplify the using process of Mask TextSpotter algorithm.

## How our project solves this problem
### Backbone models

#### Detectron2

Detectron2 is Facebook AI research’s next generation software system that implements state-of-the-art object detection algorithms. It uses a typical generalized RCNN object detection framework. 

An input image first goes through a CNN backbone to extract some image features. These features are used to predict region proposals which are regions that are likely to contain objects. The features in these regions are cropped and wrapped into some regional features, and then different types of prediction heads use regional features and image features to predict key points as well as densepose for each human found in the image.

#### Mask TextSpotter

Mask TextSpotter is a simple and smooth end-to-end learning process, which is able to achieve accurate text detection and recognition through semantic segmentation and to handle instances of irregularly shaped text.

### Our Model
In order to implement Mask TextSpotter under detectron2 framework, we devided the model into four parts: config, ROI_head, dataset, and dataloader.

For config, we rewrote config part to claim the needed setting for this task under detectron2.

In order to make Mask TextSpotter model be applicable under detectron2, we rewrote ROI_head part, which is also the backbone of this model. ROI_head can be devided into MaskHead and BoxHead to implement two tasks: global text instance segmantation task and character segmentation task. 

Since detectron2 is using dataset in coco format, while Mask TextSpotter does not have predefined dataset in that format, we need to rewrite dataloader to extract features in images and convert dataset into coco format. 

## How to train the model

## How to use the trained model
This part should be the same as original Detectron2, although our model is not completed yet.

### Installing our model
#### Requirements

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: ```pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```

#### Build Detectron2 from Source
After having the above dependencies and gcc & g++ ≥ 5, run:

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && python -m pip install -e .

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```

#### Install Pre-Built Detectron2

```
# for CUDA 10.1:
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
You can replace cu101 with "cu{100,92}" or "cpu".
```
