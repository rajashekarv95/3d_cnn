from tqdm import tqdm
import json
import numpy as np
import multiprocessing as mp
import os

# from utils.generate_anchor_boxes import generate_anchor_boxes
from .generate_anchor_boxes import generate_anchor_boxes
from .iou import get_iou
from .get_delta import get_delta

def gen_labs_parallel(idx, img_dict, anchors, mask):
    # print(idx)
    img_dict_keys = list(img_dict.keys())
    im_id = img_dict_keys[idx]

    one_labs = []
    zero_labs = []
    cat_list = []
    bb_list = []
    id_list = []
    
    one_idx = set()
    zero_idx = set()
    n_anchors = len(anchors)
    # print(img_dict[im_id])

    for annot in img_dict[im_id]:
        # print('inside')
        # print(annot)
        cat_list.append(annot['category_id'])
        bb_list.append(annot['bbox'])
        id_list.append(annot['id'])
        
        bb = annot['bbox']
        bb_ref = [bb[0], bb[0] + bb[2],bb[1],  bb[1] + bb[3]]
        # print('bb_ref', bb_ref)
        bb_broadcast = [bb_ref for _ in range(n_anchors)]
        ls = list(zip(anchors, bb_broadcast, mask))
        
        op = np.array([get_iou(ls[i]) for i in range(n_anchors)])
        # print('ious: ', sum(op))
        max_idx = int(np.argmax(op))
        
        if max_idx not in one_idx:
            one_labs.append([max_idx,get_delta(anchors[max_idx], bb_ref)])
            one_idx.add(max_idx)
        
        mask_iou = list(np.where(op >= 0.7)[0])
        n = len(mask_iou)

        # print('zero_mask_length: ', len(mask_iou))
        for i in range(n):
            if mask_iou[i] != max_idx and mask_iou[i] not in zero_idx and mask_iou[i] not in one_idx:
                # print('inside')
                one_labs.append([int(mask_iou[i]),get_delta(anchors[mask_iou[i]], bb_ref)])
                one_idx.add(mask_iou[i])
        
    for annot in img_dict[im_id]:
        bb = annot['bbox']
        bb_ref = [bb[0], bb[0] + bb[2],bb[1],  bb[1] + bb[3]]
        
        bb_broadcast = [bb_ref for _ in range(n_anchors)]
        ls = list(zip(anchors, bb_broadcast, mask))
        
        op = np.array([get_iou(ls[i]) for i in range(n_anchors)])
        
        
        mask_iou = list(np.where(op <= 0.3)[0])
        n = len(mask_iou)

        # print('zero_mask_length: ', len(mask_iou))
        for i in range(n):
            if mask_iou[i] != max_idx and mask_iou[i] not in zero_idx and mask_iou[i] not in one_idx:
                # print('inside')
                zero_labs.append(int(mask_iou[i]))
                zero_idx.add(mask_iou[i])
    print("idx: {}, Image id: {}, ones: {}, zeros: {}".format(idx, annot['image_id'], len(one_labs), len(zero_labs)))

    return ({
        'image_id': annot['image_id'], 
        'category_id': cat_list, 
        'bbox': bb_list, 
        'id': id_list,
        'one_labels': one_labs,
        'zero_labels': zero_labs
    })


def generate_labels(file_name):
    with open(file_name) as f:
        data = json.load(f)

    resolutions = [64, 128, 256, 512, 1024]
    aspect_ratios = [[1,1], [1,2], [2,1]]

    anchors, mask = generate_anchor_boxes(resolutions, aspect_ratios, steps= 16)

    n = len(anchors)
    d = []

    img_dict = {}

    for annot in data:
        if annot['image_id'] in img_dict:
            img_dict[annot['image_id']].append(annot)
        else:
            img_dict[annot['image_id']] = [annot]

    
    img_dict_keys = list(img_dict.keys())
    n_iter = len(img_dict_keys)
    # n_anchors = len(anchors)

    # for idx in tqdm(range(n_iter)):

    # gen_labs_parallel(0, img_dict)

    num_workers = int(2 * mp.cpu_count())
    print("Running on {} threads".format(num_workers))
    pool = mp.Pool(processes=num_workers)    
    # results = []
    # for idx in tqdm(range(24)):
    #     results.append(pool.apply_async(gen_labs_parallel, args=(idx, img_dict, anchors, mask)))
    results = [pool.apply_async(gen_labs_parallel, args=(idx, img_dict, anchors, mask)) for idx in tqdm(range(n_iter))]
    d = [p.get() for p in results]
    # d.append()
        # print('zeros: ', zero_idx)
        # print('ones: ', one_idx)
        # print(one_labs)
        # if idx == 100:
        #     break
    pool.close()

    output_file_name = "./coco/annotations/labels.json"
    # print(d)
    with open(output_file_name, "w") as f:
        json.dump(d, f)
    return d

def generate_clf_reg_labels(file_name):
    # file_name = "./coco/annotations/labels.json"
    if not os.path.exists("./coco/annotations/labels"):
        os.mkdir("./coco/annotations/labels")
    with open(file_name) as f:
        data = json.load(f)
    
    
    for idx in tqdm(range(len(data))):
        d = []
        label = data[idx]
        clf_labels = -1 * np.ones(shape = (39, 29, 15))
        reg_labels = np.zeros(shape = (39, 29, 15 * 4))
        # print(label['one_labels'])
        for one_lab in label['one_labels']:
            # print(one_lab)
            a = one_lab[0] // 15
            z = one_lab[0] % 15

            x = a // 29
            y = a % 29 
            # print(x,y,z)

            clf_labels[x,y,z] = 1

            for i in range(4):
                reg_labels[x,y,z*4 +i] = one_lab[1][i]
        
        for zero_lab in label['zero_labels']:
            a = zero_lab // 15
            z = zero_lab % 15

            x = a // 29
            y = a % 29 

            clf_labels[x,y,z] = 0
        # print(len(label['one_labels']))
        
        d.append({
            'image_id': label['image_id'], 
            'category_id': label['category_id'], 
            'bbox': label['bbox'], 
            'id': label['id'],
            'clf_labels': list(clf_labels.flatten()),
            'reg_labels': list(reg_labels.flatten())
        })

        output_file_name = "./coco/annotations/labels/" + str(label['image_id']) + ".json"

        with open(output_file_name, "w") as f:
            json.dump(d, f)
   
    
    # print(d)
   
     