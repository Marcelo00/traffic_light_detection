# Model for detecting and classifying traffic lights

This repository was part of the recruitment process as a working student for machine learning at UnderstandAI.

## Training
To start the training, one can start the bash script or use the trainer.py directly

## Model

The model used for training is a Faster RCNN model from Torchvision. The model predicts the bounding box and then classifies it. The score for evaluation is mAP.

## Data

I used data from <https://hci.iwr.uni-heidelberg.de/node/6132>. With the help of the script `utils/create_balanced_dataset.py`, I tried to get a balanced data set, as only a few images with bounding boxes around traffic lights are off or yellow. Therefore, I added every image with such bounding boxes and then added the remaining images. However, every traffic light class has a maximum occurrence of 1000, so the balance between the classes is stable without adding too few images for the training process.

## Possible adjustments
* a meaningful confidence measure for the prediction or generally includes different prediction scores. I only used the default prediction score - from the Pytorch model - in order to evaluate the performance of the model
* add a script for annotating images so that one can have a visual reference of how well the model is doing
* hyperparameter tuning
* different datasets
* compare with other models (-> find out a ground truth)
* tweak with the faster RCNN backbound

