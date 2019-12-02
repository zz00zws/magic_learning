import os
from PIL import Image
import numpy as np

anno_src = "./data/list_landmarks_celeba.txt"  # 关键点
anno_src2 = "./data/list_bbox_celeba.txt" #  坐标
anno_src3 = "./data/list_attr_celeba.txt" #  特征


img_dir = r"./data/img_celeba"  # 加载图片存储路径

strs=[]
with open(anno_src2) as f:
    a=f.readlines()
    for j in a:
        strs.append(j.split())
with open(anno_src) as f:
    a=f.readlines()
    for i,j in enumerate(a):
        strs[i]=strs[i]+j.split()[1:]    
        
index=[]
with open(anno_src3) as f:
    a=f.readlines()
    for j,i in enumerate(a):
        if i.split()[16]=='1':
            index.append(j)
index=index[::-1]
for i in index:
    del strs[i]           
            
for i in strs:
    image_filename = i[0].strip()
    print(image_filename)
    image_file = os.path.join(img_dir,image_filename)  # 图片存储路径连接图片名得到图片的绝对路径，通过该路径可以依次拿到文件夹下的图片，即通过样本描述txt文件里的数据拿到对应的图片

    # 获取图片后对图片进行样本增样
    with Image.open(image_file) as img:
        img_w, img_h = img.size  # 获取图片的宽和高
        x1 = float(i[1].strip())
        y1 = float(i[2].strip())
        w = float(i[3].strip())
        h = float(i[4].strip())
        x2 = x1 + w
        y2 = y1 + h

        # 人脸关键点(暂时不做)
        px1 = float(i[5].strip())  # 左眼
        py1 = float(i[6].strip())
        px2 = float(i[7].strip())  # 右眼
        py2 = float(i[8].strip())
        px3 = float(i[9].strip())  # 鼻子
        py3 = float(i[10].strip())
        px4 = float(i[11].strip())  # 左嘴角
        py4 = float(i[12].strip())
        px5 = float(i[13].strip())  # 右嘴角
        py5 = float(i[14].strip())

        # 过滤字段(排除不标准或过小的框)
        # 由于过小的样本中人脸太小，会导致训练出的网络精度较低，误框率就较高
        if max(w, h) < 40 or min(w, h) < 0 or x1 < 0 or y1 < 0:
            continue
        # 关键点不在原框内
        if px1 < x1 or py1 < y1 or px2 > x2 or py2 < y1 or px4 < x1 or py4 > y2 or px5 > x2 or py5 > y2:
            continue

        # 生成关键点最大矩形框
        landmark_max_x1 = np.minimum(px1, px4)
        landmark_max_y1 = np.minimum(py1, py2)
        landmark_max_x2 = np.maximum(px2, px5)
        landmark_max_y2 = np.maximum(py4, py5)
        landmark_max_w = landmark_max_x2 - landmark_max_x1
        landmark_max_h = landmark_max_y2 - landmark_max_y1

        # 生成人脸缩放矩形框(即新标签坐标)
        # 0.85 1 0.85 0.85
        new_x1 = landmark_max_x1 - (landmark_max_x1 - x1) * 0.85
        new_y1 = landmark_max_y1 - (landmark_max_y1 - y1) * 1
        new_x2 = landmark_max_x2 + (x2 - landmark_max_x2) * 0.85
        new_y2 = landmark_max_y2 + (y2 - landmark_max_y2) * 0.85
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1

        img = Image.open(image_file)
#        if py1<py3<py2 or py2<py3<py1 or px3>px2 or px3<px1 or w>=h or px1>px3 or px4>px3 or (px3>px4 and px3>px5) or (px3<px4 and px3<px5):  # 特殊图片

        if w < h and py1 <= py3 <= py2:  # 右斜脸
            new_x1 = x1
            flap_y1 = landmark_max_y1 - (landmark_max_y1 - y1) * 0.77
            if flap_y1 > y1:  # 不超出原框
                new_y1 = flap_y1
        if w < h and py1 >= py3 >= py2:  # 左斜脸
            new_x2 = x2
            flap_y1 = landmark_max_y1 - (landmark_max_y1 - y1) * 0.79
            if flap_y1 > y1:  # 不超出原框
                new_y1 = flap_y1
        if w < h and px3 > px2:  # 右侧脸
            continue

        if w < h and px3 < px1:  # 左侧脸
            continue

        if w < h and px3 > px4 and px3 > px5:  # 右侧脸
            continue

        if w < h and px3 < px4 and px3 < px5:  # 左侧脸
            continue

        leye_nose = px3 - px1
        reye_nose = px2 - px3
        isFront = px1 < px3 < px2 and px4 < px3 < px5 and w < h
        if isFront and reye_nose > leye_nose * 1.5:  # 正面左侧脸
            flap_x1 = landmark_max_x1 - landmark_max_w * 0.43
            flap_x2 = landmark_max_x2 + reye_nose * 1.2
            new_x2 = np.minimum(new_x2, flap_x2)
            if flap_x1 > x1:  # 预防超出原框
                new_x1 = flap_x1
        elif isFront and leye_nose > reye_nose * 1.5:  # 正面右侧脸
            flap_x2 = landmark_max_x2 + landmark_max_w * 0.43
            flap_x1 = landmark_max_x1 - leye_nose * 1.2
            new_x1 = np.maximum(new_x1, flap_x1)
            if flap_x2 < x2:  # 预防超出原框
                new_x2 = flap_x2
        elif isFront:
            flap_x1 = landmark_max_x1 - landmark_max_w * 0.61
            flap_x2 = landmark_max_x2 + landmark_max_w * 0.61
            new_x1 = np.maximum(new_x1, flap_x1)
            new_x2 = np.minimum(new_x2, flap_x2)
        if w >= h or py4 < py3 < py2 or py5 < py3 < py2:  # 水平脸
            new_y1 = y1
            new_y2 = y2

        # 调整下巴
        if isFront:  # 正面脸
            flap_y2 = landmark_max_y2 + (px5 - px4) * 0.945
            new_y2 = min(new_y2, flap_y2)

        elif w < h and py3 > np.maximum(py1, py2) and py3 < np.minimum(py4, py5) and abs(px5 - px4) < 40 and abs(
                px5 - px4) < landmark_max_h / 2.1:
            flap_y2 = landmark_max_y2 + new_h * 0.195
            new_y2 = min(new_y2, flap_y2)
        elif w < h and py3 > np.maximum(py1, py2) and py3 < np.minimum(py4, py5):
            flap_y2_1 = landmark_max_y2 + (px5 - px4) * 1.25
            flap_y2_2 = landmark_max_y2 + new_h * 0.245
            new_y2 = min(new_y2, flap_y2_1, flap_y2_2)
        elif w < h and not (py4 < py3 < py2 or py5 < py3 < py2):
            flap_y2 = landmark_max_y2 + new_h * 0.245
            new_y2 = min(new_y2, flap_y2)
        # 额头调整
        if isFront:  # 正面脸
            flap_y1 = landmark_max_y1 - landmark_max_w * 0.95
            if flap_y1 > new_y1:  # 不超出原框
                new_y1 = flap_y1
        meanx,meany=(new_x1+new_x2)/2,(new_y1+new_y2)/2
        h=max((new_x2-new_x1)/2,(new_y2-new_y1)/2)
        img_c=img.crop((int(meanx-h), int(meany-h), int(meanx+h), int(meany+h)))
        img_c=img_c.resize((96,96))
        img_c.save(os.path.join('./data/img',image_filename))
