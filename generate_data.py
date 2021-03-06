import cv2
import os
import PIL.Image as Image
from helper_functions import read_pascal,compute_iou

# this module creates licence-plate trainings data from the dataset specified in the variables data_dir
# to do this the module is creating images with licence plates and no licence plates and saves them in the folders accordingly (see car_save_path and no_car_save_path)

def main():
    car_save_path = 'use_data_only_eu/plate/'
    no_car_save_path = 'use_data_only_eu/not_plate/'

    data_dir = 'data_eu_only/'

    total_not_car = 0
    total_car = 0
    for file in os.listdir(data_dir):
        if '.png' in file:
            name,box_list = read_pascal(data_dir+file.split('.')[0]+'.xml')
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            pic = cv2.imread(data_dir+file)
            pic_copy = pic.copy()

            ss.setBaseImage(pic)
            ss.switchToSelectiveSearchFast()
            results = ss.process()

            car_count = 0
            no_car_count = 0
            total_counted = 0
            for found_box in results:
                found_box_use = [found_box[0],found_box[1],found_box[0]+found_box[2],found_box[1]+found_box[3]]
                image_roi = pic_copy[found_box[1]:found_box[3]+found_box[1],found_box[0]:found_box[0]+found_box[2]]
                iou = compute_iou(found_box_use,box_list[0]) #its a nested list, so we take the 1st element
                print(iou)
                if iou>0.7:
                    if car_count < 16: #don't have enough memory for too many

                        image_roi_use = cv2.resize(image_roi,(128,128))
                        image_roi_use = Image.fromarray(image_roi_use)
                        image_roi_use.save(car_save_path+'plate_'+str(total_car)+'.png')

                        total_car += 1
                        car_count+=1

                if iou<0.3:
                    if no_car_count < 16:
                        image_roi_use = cv2.resize(image_roi,(128,128))
                        image_roi_use = Image.fromarray(image_roi_use)
                        image_roi_use.save(no_car_save_path+'not_plate_'+str(total_not_car)+'.png')

                        #X_no_car.append(image_roi)

                        #print(no_car_save_path+'not_car_'+str(total_not_car)+'.png')
                        total_not_car+=1
                        no_car_count+=1

                if total_counted > 999:
                    break

                total_counted+=1

main()
