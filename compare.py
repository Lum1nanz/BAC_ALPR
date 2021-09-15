import cv2
import os

# this file compares all the different approaches for LP-detection
# 1) we try our conventional LP-detection (No ML or AI) (filename_1)
# 2) our LP-Detection with the slowCNN (filename_2)
# 3) LP-Detection with YOLO (filename_3)
# 4) filter Car + conventional LP-detection (filename_4)
# 5) filter Car + LP-detection with slowCNN (filename_5)
# 6) filter car + LP-detection with YOLO (filename_6)
# results are saved in one folder with different names (see brackets above)
# based on the results the different apporaches get points:
# 0: no LP detected
# +1: LP is detected without other elements
# +2: only LP is detected

#conventional approach
from cut_out_licence_plate import getLicencePlate

# slowCNN
from extract import extract_LP
from tensorflow.keras.models import load_model
model_eu_15 = load_model('models/model_eu_only_15epochs.h5')

# YOLO-LP
from yolo_lp_detector import detect_lp, initClasses, initNet
yolo_lp_net = initNet('models/yolo_lp.weights', 'models/yolo_lp.cfg')
yolo_lp_classes = initClasses('models/yolo_lp.names')

# this is the car-filter
from car_detector import detect_car
yolo_car_net = initNet('models/yolo_car.weights','models/yolo_car.cfg')
yolo_car_classes = initClasses('models/coco.names')

datadir = 'testdata/'
testdata = os.listdir(datadir)
print(testdata)

outputdir = 'compare_output/'
outputdir_content = os.listdir(outputdir)

nocar = 0
piccount = 0
# go through all pictures in the directory
for pic in testdata:
    piccount += 1
    picname = pic.split('.')[0]

    # check if the picture is already present in the output folder
    if(picname + "_0.png" in outputdir_content):
        continue

    original_img = cv2.imread(datadir + pic)
    cv2.imwrite(outputdir + picname + "_0" + ".png",original_img)


    #call conventional aproach & save resulting image
    conv_img = getLicencePlate(original_img)
    cv2.imwrite(outputdir + picname + "_1" + ".png", conv_img)

    # call slowCNN & save the resulting image
    slowCNNPic = extract_LP(original_img,model_eu_15)
    cv2.imwrite(outputdir + picname + "_2" + ".png",slowCNNPic)

    # call YOLO-CNN for LP-detection
    yolo_LP_pic = detect_lp(yolo_lp_net,yolo_lp_classes,original_img)
    cv2.imwrite(outputdir + picname + "_3" + ".png", yolo_LP_pic)

    # call car_detector to filter out the car
    car_img = detect_car(yolo_car_net, yolo_car_classes, original_img)

    # if no car is found by the cnn -> next pic
    try:
        if car_img == None:
            continue
    except ValueError:
        pass

    # call conventional aproach with car_img & save resulting image
    filtered_conv_img = getLicencePlate(car_img)
    try:
        if filtered_conv_img == None:
            pass
    except ValueError:
        cv2.imwrite(outputdir + picname + "_4" + ".png", filtered_conv_img)

    # call slowCNN with car_img
    filtered_slowCNNPic = extract_LP(car_img,model_eu_15)
    cv2.imwrite(outputdir + picname + "_5" + ".png", filtered_slowCNNPic)

    # call YOLO-CNN with LP-detection with car_img
    filtered_yolo_LP_pic = detect_lp(yolo_lp_net, yolo_lp_classes, car_img)
    cv2.imwrite(outputdir + picname + "_6" + ".png", filtered_yolo_LP_pic)

print(str(nocar) + " out of " + str(piccount))