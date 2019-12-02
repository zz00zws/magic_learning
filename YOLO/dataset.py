import torch
from torch.utils.data import Dataset
import torchvision
import cfg
import os
from PIL import Image
import math

LABEL_FILE_PATH = r"./label.txt"
IMG_BASE_DIR = r"./data/Original"   

def one_hot(cls_num, v):     
    b = torch.zeros(cls_num)
    b[v-1] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH,'r') as f:
            self.dataset = f.readlines() 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index] 
        strs = line.split()
        img_tensor = torchvision.transforms.ToTensor()(Image.open(os.path.join(IMG_BASE_DIR, strs[-1]))) 
        norm_t=torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        img_data=norm_t(img_tensor)
#        print(img_data.size())
        boxes = torch.tensor(list(map(float,strs[:-1]))).view(-1,5)          

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():      #取出对应特征图的建议框
            labels[feature_size] = torch.zeros([int(cfg.IMG_HEIGHT/feature_size),int(cfg.IMG_WIDTH/feature_size), 3, 5 + cfg.CLASS_NUM]) #生成标签格式的4维0矩阵，例：（13，13，3，15），然后将正样本插入第4维度

            for box in boxes:    #取出每组标签实际框
                cls, x1, y1, x2, y2 = box
                cx=(x1+x2)/2
                cy=(y1+y2)/2
                w=x1-x2
                h=y1-y2
                cx_offset, cx_index = math.modf(cx/feature_size)     #计算中心点X坐标在原图对应的方格位置，取小数位，整数位为13x13方格索引
                cy_offset, cy_index = math.modf(cy/feature_size)     #计算中心点Y坐标在原图对应的方格位置，取小数位，整数位为13x13方格索引

                for i, anchor in enumerate(anchors):      #依次取出三个建议框
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]    #根据索引取出三个建议框的面积
                    p_w, p_h = w / anchor[0], h / anchor[1]             #w和h分别除以建议框的w，h得到缩放比例
                    p_area = w * h                                      #实际框的面积
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)         #计算实际框和建议框的面积iou比
                    labels[feature_size][int(cy_index), int(cx_index), i] = torch.tensor(            #将得到的偏移量插入上面的四维矩阵中，得到成样本，其余的为负样本
                        [iou, cx_offset, cy_offset, torch.log(p_w), torch.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])

        return labels[32], labels[16], labels[8], img_data

























