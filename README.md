# Model for detecting and classifying traffic lights

## Training
In order to start the training one can start the bash script or use the trainer.py directly

## Model

The model used for training is a Faster RCNN model from torchvision. The model predicts the bounding box and then classifies. Therefore, the model only get the image for the evaluation. The score is for evaluation is mAP.

## Data

I used data from <https://hci.iwr.uni-heidelberg.de/node/6132>. With help of the script `utils/create_balanced_dataset.py`. I tried to get a balanced data_set as there are only few images with bounding boxes aroung traffic lights being off or yellow. Therefore, I firstly added every image with such bounding boxes and then added remaing images. However, every traffic light class has a maximum occurence of 1000 so that the balance between the classes is stable without adding to few images for the training process.

## Possible adjustments
* a meaningful confidence measure for the prediciton or in general include different prediction scores. I only used the default predition score - from the pytorch model - in order to evaluate the performance of the model
* add a script for annotating images so that one can have a visual reference of how well the model is doing
* hyperparameter tuning
* different datasets
* compare with other models (-> find out a ground truth)
* tweak with the faster rcnn backbound

