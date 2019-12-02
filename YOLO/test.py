import cfg
import PIL.Image as pimg
import torch
import os
import PIL.ImageDraw as draw
import utils as ut
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm_t=torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
IMG_BASE_DIR = r"./data/Original"   
path_out = r"./img_out"

def cmt(x,num,p,w,h):
    x=x.permute(0,3,2,1)#nwhc
    x=x[0].view(int(w/num),int(h/num),3,5+cfg.CLASS_NUM)
    c=torch.nonzero(torch.gt(x[:,:,:,0],p)).float()
    zxd = x[:,:,:,0][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    cx = x[:,:,:,1][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    cy = x[:,:,:,2][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    w = x[:,:,:,3][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    h = x[:,:,:,4][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    if c.size(0)==0:
        s=torch.tensor([]).view(c.size(0),0).to(device)
    else:
        s=torch.tensor([]).view(c.size(0),-1).to(device)
    for i in range(cfg.CLASS_NUM):
        ss = x[:,:,:,5+i][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
        s=torch.cat((s,ss),1)
    if s.size(0)==0:
        return torch.tensor([]).view(-1,6).to(device)
    a,s=torch.max(s,dim=1)
    s=s.view(-1,1).float()
    tk=torch.tensor(cfg.ANCHORS_GROUP[num])[c[:,2].cpu().numpy().tolist()].to(device).float()*torch.exp(torch.cat((w,h),1))
    out=torch.cat((s,zxd,c[:,0:1]*num+num*cx-tk[:,:1]/2,c[:,1:2]*num+num*cy-tk[:,1:]/2,
                   c[:,0:1]*num+num*cx+tk[:,:1]/2,c[:,1:2]*num+num*cy+tk[:,1:]/2),1)
    return out
        
def test(a):
    net = torch.load(r'./net.pth').to(device)
    net.eval()
    with torch.no_grad():
        for i in os.listdir(IMG_BASE_DIR):
            print(i)
            img=pimg.open(os.path.join(IMG_BASE_DIR,i))
            w,h=img.size
            im=torchvision.transforms.ToTensor()(img)
            im=norm_t(im).view(1,3,h,w).to(device)
            out_13,out_26,out_52=net(im)#nchw
            out_all=ut.nms(torch.cat((cmt(out_13,32,0.5,w,h),cmt(out_26,16,0.5,w,h),cmt(out_52,8,0.5,w,h)),0)).int().cpu().detach().numpy().tolist()
            img_draw = draw.ImageDraw(img)
            for j in out_all:
                img_draw.rectangle(j[1:5],fill=None,outline=cfg.color[j[5]-1])
            img.save('{}/{}.png'.format(path_out,str(i)+str(a)))
    
if __name__ == '__main__':
    test(0)
    
    
    
    




































