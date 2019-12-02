import torch
import os
import PIL.Image as pimg
import numpy as np

path='./data/SegmentationClass'
path2='./data/JPEGImages'
savepath='./data/label'

for name in os.listdir(path):
    img=pimg.open(os.path.join(path,name))
    x,y=img.size
    n,m=x//16,y//16
    img=img.resize((n*16,m*16))
    img.save(os.path.join('./data/label',name))
    img=pimg.open(os.path.join(path2,name.split('.')[0]+'.jpg'))
    img=img.resize((n*16,m*16))
    img.save(os.path.join('./data/img',name))
    print(name)











































