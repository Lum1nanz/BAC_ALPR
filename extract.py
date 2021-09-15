import cv2
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import load_data,non_max_suppression_fast
from tensorflow.keras.models import load_model

# this module is for the R-CNN to extract the licence plate from the image given

def extract_LP(image, model, model_name=""):
    original_img = image
    h, w, _ = original_img.shape

    new_height = 480
    compute_img = cv2.resize(original_img, (int(w/h*new_height), new_height))
    nh, nw, _ = compute_img.shape

    image = compute_img
    image = compute_img[int(0.1*nh) : int(0.9*nh), int(0.1*nw) : int(0.9*nw)]

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast(base_k=550, inc_k=550)
    #ss.switchToSelectiveSearchFast()
    results = ss.process()
    copy = image.copy()
    copy2 = image.copy()
    positive_boxes = []
    probs = []
    probs_dict = {}

    print("Process boxes: ", len(results))

    for box in results:
        x1 = box[0]
        y1 = box[1]
        x2 = box[0]+box[2]
        y2 = box[1]+box[3]

        roi = image.copy()[y1:y2,x1:x2]
        roi = cv2.resize(roi,(128,128))
        roi_use = roi.reshape((1,128,128,3))

        prob = float(model.predict(roi_use)[0])
        if prob > 0.8:
            positive_boxes.append([x1,y1,x2,y2])
            probs.append(prob)
            probs_dict[str(x1)+str(y1)+str(x2)+str(y2)] = prob
            #cv2.rectangle(copy2,(x1,y1),(x2,y2),(255,0,0),5)

    cleaned_boxes = non_max_suppression_fast(np.array(positive_boxes),0.1,probs)
    total_boxes = 0
    for clean_box in cleaned_boxes:
        clean_x1 = clean_box[0]
        clean_y1 = clean_box[1]
        clean_x2 = clean_box[2]
        clean_y2 = clean_box[3]
        total_boxes+=1
        cv2.rectangle(copy,(clean_x1,clean_y1),(clean_x2,clean_y2),(0,255,0),3)
        prob = probs_dict[str(clean_x1)+str(clean_y1)+str(clean_x2)+str(clean_y2)]
        cv2.putText(copy, str(prob), (clean_x1, clean_y1 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


        """
        y1 = int((clean_y1 + 0.1*nh) / nh * h)
        y2 = int((clean_y2 + 0.1*nh) / nh * h)
        x1 = int((clean_x1 + 0.1*nw) / nw * w)
        x2 = int((clean_x2 + 0.1*nw) / nw * w)
        cropped = original_img[y1 : y2, x1: x2]

        cv2.imshow("cropped", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    return copy
