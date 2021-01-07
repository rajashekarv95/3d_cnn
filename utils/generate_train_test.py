import numpy as np
import json
import os

def generate_train_test(label_directory):
    # directory = './coco/annotations/labels/'
    image_ids = []
    for filename in os.listdir(label_directory):
        if filename.endswith(".json"):
            # print(filename)
            im_id = str(filename)[:-5]
            # print(int(im_id))
            image_ids.append(im_id)
    image_ids = np.array(image_ids)

    np.random.seed(0)
    np.random.shuffle(image_ids)

    n = len(image_ids)
    train_idx = int(0.9 * n) 
    # print(train_idx)
    train_ids = {"id": list(image_ids[:train_idx])}
    val_ids = {"id": list(image_ids[train_idx:])}

    output_file_name = './coco/annotations/train_ids.json'
    with open(output_file_name, "w") as f:
        json.dump(train_ids, f)

    output_file_name = './coco/annotations/val_ids.json'
    with open(output_file_name, "w") as f:
        json.dump(val_ids, f)

