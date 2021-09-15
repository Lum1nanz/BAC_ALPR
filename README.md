# BAC_ALPR

This repo contains a simple ALPR (Automatic Licence Plate Recogintion) that was used in my bachelor-thesis.

## Content
Inside this repo you can find a bunch of different approaches to do ALPR.
The different approaches are compared inside `compare.py` which was the main purpose of this project.

## Car-Recognition
One approach to do the ALPR was to recognise the car first and then try to extract the licence plate. The car-detection was done with a pretrained-model from YOLOv4. The licence plate-recognition was then done with a self-trained model using a dataset that is linked in the document and YOLOv4 as a framework.
