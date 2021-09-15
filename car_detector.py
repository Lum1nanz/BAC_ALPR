import cv2
import numpy as np

def detect_car(net, classes, image):

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    H,W = image.shape[0:2]

    blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.6:
                centerX = int(detection[0] * W)
                centerY = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x,y,int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idx = cv2.dnn.NMSBoxes(boxes,confidences,0.6,0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    largest_car = None
    image_pixels = 0
    for i in range(len(boxes)):
        if i in idx:
            x,y,w,h = boxes[i]
            label = str(classes[classIDs[i]])
            if label == "car":
                if image_pixels < h*w:
                    largest_car = image[y:y+h,x:x+w]
                    image_pixels = h*w
            # cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),2)
            # cv2.putText(image,label,(x,y+30),font,1,(255,255,255),2)
            # conf = str(confidences[i])
            # cv2.putText(image,conf,(x,y+60),font,1,(255,255,255),2)

    return largest_car

def car_detection(image):
    weights_Path = "darknet_files/yolov3.weights"
    config_Path = "darknet_files/yolov3.cfg"

    net = cv2.dnn.readNetFromDarknet(config_Path, weights_Path)
    classes = []
    with open("darknet_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    car = detect_car(net, classes, image)

    if car is None:
        exit(1)

    # feed the detected car to the licence plate recognition

    return car

def initNet(weights_path, config_path):
    return cv2.dnn.readNetFromDarknet(config_path, weights_path)

def initClasses(classes_path):
    classes = []
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

if __name__ == '__main__':
    image = cv2.imread('pictures/24.png')
    car_detection(image)