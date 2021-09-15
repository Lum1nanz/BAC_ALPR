import cv2
import numpy as np

# this module is loading the YOLO-Model to detect the licence plates in the picture given

def detect_lp(net, classes, image):

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
            if confidence > 0.2:
                centerX = int(detection[0] * W)
                centerY = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x,y,int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    CONFIDENCE_THRESHHOLD = 0.2
    IOU_TRESHHOLD = 0.2

    idx = cv2.dnn.NMSBoxes(boxes,confidences,CONFIDENCE_THRESHHOLD,IOU_TRESHHOLD)

    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in idx:
            x,y,w,h = boxes[i]
            label = str(classes[classIDs[i]])

            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(image,label,(x,y+30),font,1,(0,255,0),2)
            conf = str(confidences[i])
            cv2.putText(image,conf,(x,y+60),font,1,(0,255,0),2)
    return image

def initNet(weights_path, config_path):
    return cv2.dnn.readNetFromDarknet(config_path, weights_path)

def initClasses(classes_path):
    classes = []
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

if __name__ == '__main__':
    image = cv2.imread('pictures/24.png')
    net = initNet('models/yolo_lp.weights', 'models/yolo_lp.cfg')
    classes = initClasses('models/yolo_lp.names')
    YOLO_LP_Img = detect_lp(net,classes,image)
