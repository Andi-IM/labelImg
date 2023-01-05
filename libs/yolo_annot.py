import glob
import os
import cv2
from libs.annoter.annot_utils import save_yolo, get_BBoxYOLOv7
from libs.annoter.hubconf import custom


class YoloAnnotater:

    def __init__(self, dataset, model, size, confidence):
        self.path_to_dir = dataset
        self.path_or_model = model
        self.img_size = size
        self.detect_conf = confidence

    def detect(self):
        model = custom(path_or_model=self.path_or_model)

        img_list = glob.glob(os.path.join(self.path_to_dir, '*.jpg')) + \
            glob.glob(os.path.join(self.path_to_dir, '*.jpeg')) + \
            glob.glob(os.path.join(self.path_to_dir, '*.png'))

        for img in img_list:
            folder_name, file_name = os.path.split(img)
            image = cv2.imread(img)
            h, w, c = image.shape
            bbox_list, class_list, confidence = get_BBoxYOLOv7(image, model, self.detect_conf)
            save_yolo(folder_name, file_name, w, h, bbox_list, class_list)

            print(f'Successfully Annotated {file_name}')
        print('YOLOv7-Auto_Annotation Successfully Completed')
