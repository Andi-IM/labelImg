import os


# Function to convert YOLO (.txt) format
def save_yolo(folder_name, file_name, w, h, bbox_list, class_list):
    txt_name = os.path.splitext(file_name)[0] + '.txt'
    path_to_save = os.path.join(folder_name, txt_name)
    out_file = open(path_to_save, 'w')
    for box, class_index in zip(bbox_list, class_list):
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]

        x_center = float((x_min + x_max)) / 2 / w
        y_center = float((y_min + y_max)) / 2 / h

        width = float((x_max - x_min)) / w
        height = float((y_max - y_min)) / h

        # Save
        out_file.write("%d %.6f %.6f %.6f %.6f\n" %
                       (int(class_index-1), x_center, y_center, width, height))

    print(f'Successfully Created {txt_name}')


def get_BBoxYOLOv7(img, yolo_model, detect_conf):

    # Load YOLOv7 model on Image
    results = yolo_model(img)

    # Bounding Box
    box = results.pandas().xyxy[0]
    bbox_list = []
    confidence = []
    class_ids = []
    # Class
    class_list = box['class'].tolist()
    # save_yolo function need class index starting from 1 NOT Zero
    new_list = [x+1 for x in class_list]

    for i, id in zip(box.index, new_list):
        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][i]

        # detect_conf
        if conf > detect_conf:
            # BBox
            bbox_list.append([xmin, ymin, xmax, ymax])
            # class
            class_ids.append(id)
            # Confidence
            confidence.append(conf)
    return bbox_list, class_ids, confidence
