import json

from utils.refine_annotations import refine_annotations
from utils.refine_annotations import generate_refined_dataset
from utils.refine_annotations import refine_bounding_box 
from coco.models.rcnn import rcnn

from utils.iou import get_iou
import multiprocessing as mp
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import copy

# refine_annotations(file_name="coco/annotations/instances_train2017.json", num_categories = 5)

# generate_refined_dataset(file_name = './coco/annotations/refined_annotations.json', image_folder= './coco/train2017/')

# refine_bounding_box(file_name = './coco/annotations/refined_annotations.json', image_folder= './coco/train2017/')

# file_name = "./coco/annotations/refined_annotations.json"
# with open(file_name) as f:
#     data = json.load(f)

# for annot in data

from utils.generate_labels import generate_labels
from utils.generate_labels import generate_clf_reg_labels
from tqdm import tqdm



def get_labels(file_name):
    with open(file_name) as f:
        data = json.load(f)
    image_folder = './coco/train2017/refined_train/'
    n = len(data['id'])
    img_array = []
    clf_labels = []
    reg_labels = []
    for i in tqdm(range(n)):
        ids = str(data['id'][i])

        file_name = image_folder + ''.join(['0' for _ in range(12 - len(ids))]) + ids + '.jpg'
        img = Image.open(file_name)
        img = np.array(img)
        if len(img.shape) != 3:
            img = img.reshape(-1,480, 640)
            img = np.vstack((img,img,img))
        else:
            img = np.moveaxis(img, 2, 0)

        img = np.moveaxis(img, 2, 1)

        assert img.shape == (3,640,480)

        img_array.append(img)

        file_name = './coco/annotations/labels/' + str(ids) + '.json'
        with open(file_name) as f:
            label_data = json.load(f)
            clf_labels.append(np.array(label_data[0]['clf_labels']).reshape(39, 29, 15))

            reg_labels.append(np.array(label_data[0]['reg_labels']).reshape(39, 29, 60))
        if i == 1000:
            break


    img = np.array(img_array)
    clf = np.array(clf_labels)
    reg = np.array(reg_labels)

    return img, clf, reg

def get_mask_for_mini_batch(x):
    out = x.view(-1)
    nz = torch.nonzero(out==1).view(-1)

    res = torch.zeros(out.shape)
    m = torch.randperm(len(nz))[0:max(len(nz), 128)]
    indices = nz[m]

    res[indices] = 1

    mask_clf_1 = out * res
    mask_clf_1 = mask_clf_1.view(x.shape)
    # print(mul_out.shape)

    mask_reg_1 = mask_clf_1.repeat_interleave(repeats = 4, dim = 1)
    # print(mul_out_1.shape)

    nz = torch.nonzero(out==0).view(-1)
    res = torch.zeros(out.shape)
    m = torch.randperm(len(nz))[0:max(len(nz), 128)]
    indices = nz[m]

    res[indices] = 1

    mask_clf_0 = out * res
    mask_clf_0 = mask_clf_0.view(x.shape)
    # print(mul_out.shape)

    mask_reg_0 = mask_clf_0.repeat_interleave(repeats = 4, dim = 1)

    return (mask_clf_0 + mask_clf_1, mask_reg_0 + mask_reg_1)

def get_mask_for_validation(x):
    out = torch.tensor(x).contiguous().view(-1)
    nz = torch.nonzero(out==-1).view(-1)
    mask_clf = torch.ones(out.shape)

    mask_clf[nz] = 0

    mask_clf = mask_clf.view(x.shape)
    mask_reg = mask_clf.repeat_interleave(repeats = 4, dim = 1)

    return mask_clf, mask_reg



if __name__ == '__main__':
    if not os.path.exists("./coco/models"):
        os.mkdir("./coco/models")
    if not os.path.exists("./coco/models/vgg16.pth"):
        vgg16 = models.vgg16(pretrained=True)
        torch.save(vgg16, "./coco/models/vgg16.pth")
    else:
        vgg16 = torch.load("./coco/models/vgg16.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vgg16_t = torch.nn.Sequential(*list(vgg16.children())[0][:-1])
    vgg16_t.to(device)
    vgg16_t.eval()
    # print(vgg16_t)

    model = rcnn()
    model.to(device)

    file_name = './coco/annotations/train_ids.json'
    train_img, train_clf, train_reg = get_labels(file_name)
    print('Loaded train data')

    train_clf = np.moveaxis(train_clf, 3, 1)
    train_reg = np.moveaxis(train_reg, 3, 1)

    train_img = torch.tensor(train_img, dtype = torch.float32).to(device)
    train_clf = torch.tensor(train_clf, dtype = torch.float32).to(device)
    train_reg = torch.tensor(train_reg, dtype = torch.float32).to(device)

    file_name = './coco/annotations/val_ids.json'
    val_img, val_clf, val_reg = get_labels(file_name)
    print('Loaded validation data')

    val_clf = np.moveaxis(val_clf, 3, 1)
    val_reg = np.moveaxis(val_reg, 3, 1)

    mask_clf_val, mask_reg_val = get_mask_for_validation(val_clf)

    val_img = torch.tensor(val_img, dtype = torch.float32).to(device)
    val_clf = torch.tensor(val_clf, dtype = torch.float32).to(device)
    val_reg = torch.tensor(val_reg, dtype = torch.float32).to(device)

    mask_clf_val = mask_clf_val.to(device)
    mask_reg_val = mask_reg_val.to(device)

    val_clf = val_clf * mask_clf_val
    val_reg = val_reg * mask_reg_val

    val_img = torch.tensor(val_img, dtype = torch.float32)
    val_clf = torch.tensor(val_clf, dtype = torch.float32)
    val_reg = torch.tensor(val_reg, dtype = torch.float32)

    # print(train_img.shape)
    # print(train_clf.shape)
    # print(train_reg.shape)

    train_dataset = TensorDataset(train_img, train_clf, train_reg)
    data_loader = DataLoader(train_dataset, batch_size= 5, shuffle= True)

    optimizer = optim.Adam(model.parameters())
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()


    for epoch in range(5):
        train_loss = 0
        for batch in data_loader:

            optimizer.zero_grad()

            img = batch[0]
            label_clf = batch[1]
            label_reg = batch[2]

            output_clf, output_reg = model(vgg16_t(img))

            mask_clf, mask_reg = get_mask_for_mini_batch(label_clf)
            mask_clf = torch.tensor(mask_clf).to(device)
            mask_reg = torch.tensor(mask_reg).to(device)
            
            # print('mask_clf: ', mask_clf.shape)
            # print('label_clf: ', label_clf.shape)
            # print('mask_reg: ', mask_reg.shape)
            # print('label_reg: ', label_reg.shape)
            
            output_clf = output_clf * mask_clf
            output_reg = output_reg * mask_reg

            label_clf = label_clf * mask_clf
            label_reg = label_reg * mask_reg



            loss = mse_loss(output_reg, label_reg) + bce_loss(output_clf, label_clf.detach())

            loss.backward()
            train_loss += loss.detach().item()
        
        model.eval()
        with torch.no_grad():
            val_output_clf, val_output_reg = model(vgg16_t(val_img))
            val_loss = mse_loss(val_output_reg, val_reg) + bce_loss(val_output_clf, val_clf.detach())
        print('Epoch Number: {}\t Train Loss: {} \t Validation Loss: {} '.format(epoch, train_loss/1000, val_loss.detach().item()/287))
        model.train()




        




