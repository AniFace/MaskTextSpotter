# MaskTextSpotter
## Main problem that our project is targeting
Given multiple object detection algorithms implemented in detectron2, and the importance of OCR problems, we want to implement Mask TextSpotter algorithm aiming at OCR task under detectron2 framework to simplify the using process of Mask TextSpotter algorithm.

## How our project solves this problem
### Backbone models

#### Detectron2

[Detectron2](https://github.com/facebookresearch/detectron2) is Facebook AI research’s next generation software system that implements state-of-the-art object detection algorithms. It uses a typical generalized RCNN object detection framework. 

An input image first goes through a CNN backbone to extract some image features. These features are used to predict region proposals which are regions that are likely to contain objects. The features in these regions are cropped and wrapped into some regional features, and then different types of prediction heads use regional features and image features to predict key points as well as densepose for each human found in the image.

#### MaskTextSpotter

[MaskTextSpotter](https://github.com/MhLiao/MaskTextSpotter) is a simple and smooth end-to-end learning process, which is able to achieve accurate text detection and recognition through semantic segmentation and to handle instances of irregularly shaped text.

### Our Model
In order to implement Mask TextSpotter under detectron2 framework, we devided the model into four parts: config, ROI_head, dataset, and dataloader.

For config, we rewrote config part to claim the needed setting for this task under detectron2.

In order to make Mask TextSpotter model be applicable under detectron2, we rewrote ROI_head part, which is also the backbone of this model. ROI_head can be devided into MaskHead and BoxHead to implement two tasks: global text instance segmantation task and character segmentation task. 

Since detectron2 is using dataset in coco format, while Mask TextSpotter does not have predefined dataset in that format, we need to rewrite dataloader to extract features in images and convert dataset into coco format. 

## How to train the model
This part should be the same as [original Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md), although our model is not completed yet.

### Training & Evaluation in Command Line

To train a model with "train_net.py", run:

```
python tools/train_net.py --num-gpus 8 \
	--config-file configs/Base-MaskTextSpotter_RCNN_FPN.yaml
```

The configs are made for 8-GPU training. To train on 1 GPU, change the batch size with:
```
python tools/train_net.py \
	--config-file configs/Base-MaskTextSpotter_RCNN_FPN.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```
For this model, CPU training is not supported.

To evaluate a model's performance, use:

```
python tools/train_net.py \
	--config-file configs/Base-MaskTextSpotter_RCNN_FPN.yaml \
	--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## How to use the trained model
This part should be the same as [original Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md), although our model is not completed yet.

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

#### Inference Demo with Pre-trained Models

1. Pick Mask TextSpotter model, which is Base-MaskTextSpotter_RCNN_FPN.yaml.
2. Detectron2 provide demo.py that is able to run builtin standard models. Run it with:
```
python demo/demo.py --config-file configs/Base-MaskTextSpotter_RCNN_FPN.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS Base-MaskTextSpotter_RCNN_FPN.pkl
```
The configs are made for training, therefore we need to specify model weights to a model for evaluation. This command will run the inference and show visualizations in an OpenCV window. (This is not implemented yet.)

For details of the command line arguments, see ```demo.py -h``` or look at its source code to understand its behavior. Some common arguments are:

- To run on your webcam, replace ```--input files``` with ```--webcam```.
- To run on a video, replace ```--input files``` with ```--video-input video.mp4```.
- To run on cpu, add ```MODEL.DEVICE cpu``` after ``--opts```.
- To save outputs to a directory (for images) or a file (for webcam or video), use ```--output```.
