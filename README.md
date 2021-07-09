# Airbus-Ship-Detection-with-MaskRCNN
Using the famous Airbus Ship Detection dataset from the corresponding Kaggle Competition,
trained a model with MaskRCNN from matterport to perform image segmentation for ship detection.
## Info
This repository is part of my senior thesis '*Alcyone* Object & Phenomenon Detection System'.

### Saronic Gulf, Greece Agia Zoni II oil spill
![](https://github.com/codelover96/Airbus-Ship-Detection-with-MaskRCNN/blob/main/results/2.png)
## Installation
1. Clone Mask RCNN repository from Matterport
2. Install it's dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Optionally Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. Clone this repo and place below files to Mask RCNN repo, inside folder `Mask_RCNN-master` 
   * [ship_obj_detection.py](Mask_RCNN-master/ship_obj_detection.py)
   * [run_ship_model.py](Mask_RCNN-master/run_ship_model.py)
5. Be sure to adjust config.py accordingly (number of classes, gpu count)
6. Create an `images` folder inside `Mask_RCNN-master` folder and copy your images to run the model on.
    
## Run
1.  In line 23 of [ship_obj_detection.py](Mask_RCNN-master/ship_obj_detection.py) point to your model h5 file.
2.  In line 25 of [ship_obj_detection.py](Mask_RCNN-master/ship_obj_detection.py) point to your test image.
3.  Same for [run_ship_model.py](Mask_RCNN-master/run_ship_model.py) file.
4.  run commands:
    * `python3 ship_obj_detection.py`
    * `python3 run_ship_model.py`
## Details
* [ship_obj_detection.py](Mask_RCNN-master/ship_obj_detection.py) - Segmentation and Detection of ships
* [run_ship_model.py](Mask_RCNN-master/run_ship_model.py) - Only detections and bounding boxes.

## Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/shapes/train_shapes.ipynb`, `samples/coco/coco.py`, `samples/balloon/balloon.py`, and `samples/nucleus/nucleus.py` of
original Mask RCNN repo [here](https://github.com/matterport/Mask_RCNN)

### Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

### TODO
-[x] added source files for running segmentation and object detection
-[ ] add trained model files

### References
* [Original Mask RCNN repo](https://github.com/matterport/Mask_RCNN)
* [Useful notebooks using Mask RCNN](https://github.com/abhinavsagar/kaggle-notebooks)
* [Deep Learning for Ship Detection and Segmentation](https://towardsdatascience.com/deep-learning-for-ship-detection-and-segmentation-71d223aca649)

##### For educational purposes
>~codelover96
