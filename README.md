# MaskTextSpotter
## Main problem that our project is targeting:
Given multiple object detection algorithms implemented in detectron2, and the importance of OCR problems, we want to implement Mask TextSpotter algorithm aiming at OCR task under detectron2 framework to simplify the using process of Mask TextSpotter algorithm.

## How our project solves this problem:
### Backbone model:

Detectron2:

Detectron2 is Facebook AI research’s next generation software system that implements state-of-the-art object detection algorithms. It uses a typical generalized RCNN object detection framework. An input image first goes through a CNN backbone to extract some image features. These features are used to predict region proposals which are regions that are likely to contain objects. The features in these regions are cropped and wrapped into some regional features, and then different types of prediction heads use regional features and image features to predict key points as well as densepose for each human found in the image.

Mask TextSpotter:

Mask TextSpotter is a simple and smooth end-to-end learning process, which is able to achieve accurate text detection and recognition through semantic segmentation and to handle instances of irregularly shaped text.

Our Model:
