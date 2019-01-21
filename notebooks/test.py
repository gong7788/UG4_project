from torchvision.datasets import CocoCaptions, CocoDetection
from torchvision import transforms
from pycocotools.coco import COCO
import torchvision.models as models
import torch

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/home/mappelgren/Desktop/correcting-agent/data/image_data'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
categories = coco.loadCats(coco.getCatIds())
names=[cat['name'] for cat in categories]


catIds = coco.getCatIds(catNms=['dog', 'skateboard']);
imgIds = coco.getImgIds(catIds=catIds );

img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread(img['coco_url'])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

T = torch.from_numpy(I)
print(normalize(T))
