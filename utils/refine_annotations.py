import json
import numpy as np
from collections import Counter
import os
from PIL import Image
from tqdm import tqdm



def refine_annotations(file_name, num_categories = 5):
    with open(file_name) as f:
        data = json.load(f)
    
    annotations = data['annotations']

    cats = list()

    for d in annotations:
        cats.append(d['category_id']) 

    counts = Counter(cats)
    counts_list = np.array(list(counts.items()))

    mask = np.argsort(counts_list[:,1])
    req_cat_counts = counts_list[mask][-1 - num_categories:-1,:]

    req_cats = req_cat_counts[:,0]

    max_counts = {}
    for cat in req_cats:
        max_counts[cat] = 0

    refined_annot = []
    count_allowed = 2000
    for annot in annotations:
        if annot['category_id'] in req_cats and annot['iscrowd'] == 0:
            if max_counts[annot['category_id']] > count_allowed:
                continue
            max_counts[annot['category_id']] += 1
            refined_annot.append({
                "image_id": annot['image_id'], 
                "category_id": annot['category_id'],
                "bbox": annot['bbox'],
                "id": annot['id']})

    output_file_name = "./coco/annotations/refined_annotations.json"

    with open(output_file_name, "w") as f:
        json.dump(refined_annot, f)
    print("File written")

def generate_refined_dataset(file_name, image_folder, v_op = 480, h_op = 640):
    with open(file_name) as f:
        data = json.load(f)
    ctr = 0

    if not os.path.exists(image_folder + '/refined_train'):
        os.makedirs(image_folder + '/refined_train')

    s = set()
    for annot in data:
        s.add(annot['image_id'])
    s = list(s)

    for i in tqdm(range(len(s))):
        file_name = image_folder + ''.join(['0' for _ in range(12 - len(str(s[i])))]) + str(s[i]) + '.jpg'

        if os.path.isfile(file_name): 
            ctr += 1
            img = Image.open(file_name)

            resized_img = img.resize((h_op, v_op))

            output_file_name = image_folder + 'refined_train/' + ''.join(['0' for _ in range(12 - len(str(s[i])))]) + str(s[i]) + '.jpg'
            resized_img.save(output_file_name)  
        else:
            print(i)          
            
    print("{} files written.".format(ctr))

def refine_bounding_box(file_name, image_folder, v_op = 480, h_op = 640):
    with open(file_name) as f:
        data = json.load(f)
    bb_ref = []
    n = len(data)
    for i in tqdm(range(n)):
        annot = data[i]
        file_name = image_folder + ''.join(['0' for _ in range(12 - len(str(annot['image_id'])))]) + str(annot['image_id']) + '.jpg'
        img = Image.open(file_name)

        v_img = np.array(img).shape[0]
        h_img = np.array(img).shape[1]

        bb = annot['bbox']

        v_ratio = v_op / v_img
        h_ratio = h_op / h_img

        bb_new = [bb[0] * h_ratio, bb[1] * v_ratio, bb[2] * h_ratio, bb[3] * v_ratio]

        bb_ref.append({
                "image_id": annot['image_id'], 
                "category_id": annot['category_id'],
                "bbox": bb_new,
                "id": annot['id']})

    output_file_name = "./coco/annotations/refined_bb_annotations.json"

    with open(output_file_name, "w") as f:
        json.dump(bb_ref, f)

        

            