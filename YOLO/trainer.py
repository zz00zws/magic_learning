import dataset
import nets
import torch
import os
import torch.nn as nn
import test as test
import PIL.Image as pimg
import PIL.ImageDraw as draw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loss_f1=nn.MSELoss()
#loss_f2=nn.CrossEntropyLoss()
def loss_fn(output, target, alpha):

    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

#    output_p=output[:,:,:,:,:5]
#    output_u=output[:,:,:,:,5:]
#    target_p=target[:,:,:,:,:5]
#    target_u=target[:,:,:,:,5:]
    mask_obj = target[..., 0] > 0           #最后一维0索引大于0的掩码
    mask_noobj = target[..., 0] == 0        #最后一维0索引等于0的掩码

    loss_obj = torch.mean((output[mask_obj] - target[mask_obj]) ** 2)           #正样本标签和输出做平方差损失
    loss_noobj = torch.mean((output[mask_noobj][:,0] - target[mask_noobj][:,0]) ** 2)      #负样本标签和输出做平方差损失
#    a,b1=torch.max(target_u[mask_obj],1)
#    loss_obj=loss_f1(output_p[mask_obj],target_p[mask_obj])+loss_f2(output_u[mask_obj],b1)
#    a,b2=torch.max(target_u[mask_noobj],1)
#    loss_noobj=loss_f1(output_p[mask_noobj],target_p[mask_noobj])#+loss_f2(output_u[mask_noobj],b2)
#    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    loss = alpha * loss_obj + loss_noobj
    return loss

save_path=r'./net.pth'


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    if os.path.exists(save_path):
        net = torch.load(save_path).to(device)
    else:
        net = nets.mainet().to(device)
    net.train()

    opt = torch.optim.Adam(net.parameters())
#    opt = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)
    a=-1
    while True:
#    for a in range(1):
        print(a)
        for target_13, target_26, target_52, img_data in train_loader:
            a=a+1
            target_13=target_13.to(device)
            target_26=target_26.to(device)
            target_52=target_52.to(device)
            img_data=img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13, target_13, 0.84)
            loss_26 = loss_fn(output_26, target_26, 0.96)
            loss_52 = loss_fn(output_52, target_52, 0.99)
        
            loss = loss_13 + loss_26 + loss_52
        
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())
            if a%100==0:
                torch.save(net,save_path)
                print('安排')
#        if a%100==0 and a!=0:
#            test.test(a)
        print('安排上了')
        
        
        
        
        
