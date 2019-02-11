from torchvision.datasets import CocoCaptions, CocoDetection, ImageFolder
from torchvision import transforms
from pycocotools.coco import COCO
import torchvision.models as models
import torch
import os
import json
from torchvision.transforms import ToPILImage
from IPython.display import Image
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#
# dataDir='/home/mappelgren/Desktop/correcting-agent/data/image_data'
# dataType='train2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# coco=COCO(annFile)
#
# image_net_labels = json.load(open('/home/mappelgren/Desktop/correcting-agent/data/image_data/imagenet_class_index.json'))
#
# categories = coco.loadCats(coco.getCatIds())
# names=[cat['name'] for cat in categories]
# print('coco categories: \n{}\n'.format(', '.join(names)))
#
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()

def save_images(category, coco, modifier='train'):
    """Saves all images of a specified category locally"""
    image_store_root_dir = os.path.join("/home/mappelgren/Desktop/correcting-agent/data/image_data/coco_cat", modifier)
    image_store_dir = os.path.join(image_store_root_dir, category)
    os.makedirs(image_store_dir, exist_ok=True)

    image_urls = get_category_img_urls(category, coco)
    for url in image_urls:
        image_data = io.imread(url)
        image_name = url.split('/')[-1]
        image_store_file = os.path.join(image_store_dir, image_name)
        io.imsave(image_store_file, image_data)



def get_category_img_ids(category, coco):
    """Given a specific category returns the image ids of that category"""
    catIds = coco.getCatIds(catNms=[category]);
    imgIds = coco.getImgIds(catIds=catIds );
    img = coco.loadImgs(imgIds)
    return img


def get_category_img_urls(category, coco):
    """Given a specific category returns the urls for images associated to that category"""
    img = get_category_img_ids(category)
    imgs = [i['coco_url'] for i in img]
    return imgs


def save_categories():
    dataDir='/home/mappelgren/Desktop/correcting-agent/data/image_data'
    dataType='train2014'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)

    category_names = ['teddy bear', 'apple', 'tv', 'laptop', 'kite', 'knife', 'bottle', 'dog']
    for category in category_names:
        save_images(category, coco)


def get_coco_dataset():
    coco_categories = ImageFolder('/home/mappelgren/Desktop/correcting-agent/data/image_data/coco_cat', transform=train_transform)
    coco_labels = {i:l for i,l in enumerate(coco_categories.classes)}
    return coco_categories, coco_labels

def get_coco_dataloader(batch_size):
    coco_categories, coco_labels = get_coco_dataset()
    return torch.utils.data.DataLoader(coco_categories,
        batch_size=batch_size, shuffle=true), coco_labels
