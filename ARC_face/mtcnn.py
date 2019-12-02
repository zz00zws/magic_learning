import PIL.Image as pimg
import os
import PIL.ImageDraw as draw
import utils as ut
import torch
import torchvision
import time

#path = r"./test_img"
path = r"./test_img"
path_out = r"./test_img_out"
norm_t=torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NET_p = torch.load('./pp2/net_p3.pth')
NET_r = torch.load('./pp2/net_r3.pth')
NET_o = torch.load('./pp2/net_o3.pth')

def c_s(ms):
    cx=(ms[:,1:2]+ms[:,3:4])/2
    cy=(ms[:,2:3]+ms[:,4:5])/2
    ms[:,1:2]=1.05*ms[:,1:2]-0.05*cx 
    ms[:,2:3]=1.1*ms[:,2:3]-0.1*cy
    ms[:,3:4]=0.95*ms[:,3:4]+0.05*cx
    ms[:,4:5]=0.9*ms[:,4:5]+0.1*cy
    return ms
    

def cmp(img_c,img_l,p,a):
    xy=(torch.nonzero(torch.gt(img_c,p))[:,2:]).float()
    l = img_c[:,0][torch.gt(img_c[:,0],p)].view(-1,1).float()
    x1 = img_l[:,0][torch.gt(img_c[:,0],p)].view(-1,1).float()
    y1 = img_l[:,1][torch.gt(img_c[:,0],p)].view(-1,1).float()
    x2 = img_l[:,2][torch.gt(img_c[:,0],p)].view(-1,1).float()
    y2 = img_l[:,3][torch.gt(img_c[:,0],p)].view(-1,1).float()
    l=torch.cat((l,(xy[:,1:]*2+x1*12)/a),1)
    l=torch.cat((l,(xy[:,:1]*2+y1*12)/a),1)
    l=torch.cat((l,(xy[:,1:]*2+x2*12)/a),1)
    l=torch.cat((l,(xy[:,:1]*2+y2*12)/a),1)
    return l
    
def build_n(img,k_s,l):
    if l.size()!=torch.Size([0]):
        a1=l[:,3]-l[:,1]
        a2=l[:,4]-l[:,2]
        a=torch.max(a1,a2)
        x=l[:,3]+l[:,1]
        y=l[:,4]+l[:,2]
        a=torch.div(a,2)
        x=torch.div(x,2)
        y=torch.div(y,2)
        x=x.int().cpu().numpy().tolist()
        y=y.int().cpu().numpy().tolist()
        a=a.int().cpu().numpy().tolist()
    xl=torch.tensor([]).view(-1,3).float()
    xs=torch.tensor([]).view(-1,3,k_s,k_s).float()
    if l.size()!=torch.Size([0]):
        for ss,(i,j,k) in enumerate(zip(x,y,a)):
            im=img.crop((i-k,j-k,i+k,j+k))
#            if k_s == 24:
#                im.save('{}/{}.png'.format(path_out,str(ss)))
            im=im.resize((k_s,k_s))
#            if k_s == 24:
#                im.save('{}/{}.png'.format(path_out,str(ss)))
            img_tensor = torchvision.transforms.ToTensor()(im) 
            im=norm_t(img_tensor).view(-1,3,k_s,k_s)
            il=torch.tensor([[i-k,j-k,2*k]]).float()
            xs=torch.cat((xs,im),0)
            xl=torch.cat((xl,il),0)
    return xs,xl

#def build_n(img,k_s,l):
#    if l.size()!=torch.Size([0]):
#        img=torchvision.transforms.ToTensor()(img)
#        img=norm_t(img)
#        img=img.view(1,img.size(0),img.size(1),img.size(2))
#        a1=l[:,3]-l[:,1]
#        a2=l[:,4]-l[:,2]
#        a=torch.max(a1,a2)
#        x=l[:,3]+l[:,1]
#        y=l[:,4]+l[:,2]
#        a=torch.div(a,2)
#        x=torch.div(x,2)
#        y=torch.div(y,2)
#        xl=torch.cat(((x-a).view(-1,1),(y-a).view(-1,1),(a*2).view(-1,1)),1).float()
#        x=x.int().cpu().numpy().tolist()
#        y=y.int().cpu().numpy().tolist()
#        a=a.int().cpu().numpy().tolist()
#        xs=torch.tensor([]).view(-1,3,k_s,k_s).float()
#        for i,j,k in zip(x,y,a):
#            if i<=k:
#                i=k
#            if j<=k:
#                j=k
#            im=img[:,:,(j-k):(j+k),(i-k):(i+k)]
#            imn=torch.nn.AdaptiveAvgPool2d(k_s)
#            im=imn(im)
#            xs=torch.cat((xs,im),0).float()
#    else:
#        xs=torch.tensor([]).view(-1,3,k_s,k_s).float()
#        xl=torch.tensor([]).view(-1,3).float()
#        
#    return xs,xl
        
def ump(loc,out_c,out_l,k_s,p):
    
    c = out_c[:,0][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x1 = out_l[:,0][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y1 = out_l[:,1][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x2 = out_l[:,2][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y2 = out_l[:,3][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x3 = out_l[:,4][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y3 = out_l[:,5][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x4 = out_l[:,6][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y4 = out_l[:,7][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x5 = out_l[:,8][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y5 = out_l[:,9][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x6 = out_l[:,10][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y6 = out_l[:,11][torch.gt(out_c[:,0],p)].view(-1,1).float()
    x7 = out_l[:,12][torch.gt(out_c[:,0],p)].view(-1,1).float()
    y7 = out_l[:,13][torch.gt(out_c[:,0],p)].view(-1,1).float()
    loc = loc[torch.gt(out_c[:,0],p)]
    c=torch.cat((c,(loc[:,:1]+x1*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y1*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,:1]+x2*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y2*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,:1]+x3*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y3*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,:1]+x4*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y4*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,:1]+x5*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y5*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,:1]+x6*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y6*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,:1]+x7*loc[:,2:]).int().float()),1)
    c=torch.cat((c,(loc[:,1:2]+y7*loc[:,2:]).int().float()),1)
    return c
        
def testp(img,NET_p):
    alph=0.7
    cms=torch.tensor([]).view(-1,5).to(device)#(-1,15)
    x,y = img.size  
    while True:
        
        img_c = img.resize((int(x*alph),int(y*alph)))
        img_tensor = torchvision.transforms.ToTensor()(img_c)
        inp=norm_t(img_tensor).view(1,3,int(y*alph),int(x*alph)).to(device)
        out_p_c,out_p_l=NET_p(inp)#out(1,5,n,m)
        cm=ut.nms(cmp(out_p_c,out_p_l[:,:4],0.7,alph),5,0.4).to(device)#(-1,5)
        cms=torch.cat((cms,cm),0)
        alph=alph*0.7
        if alph*img.size[0]<=12 or alph*img.size[1]<=12:
            break
        
#    cms=ut.nms(cms,0.99,True)
#    print('p',cms.size(0))
    return cms

def testr(img,cms,NET_r):
    if cms.size()==torch.Size([0,5]):
        return torch.tensor([]).view(-1,15) 
    inp_r,loc_r=build_n(img,24,cms)
    loc_r=loc_r.to(device)
    inp_r=inp_r.to(device)
    out_r_c,out_r_l=NET_r(inp_r)#out(-1,5)p
    rms=ut.nms(ump(loc_r,out_r_c,out_r_l,24,0.9),15,0.9).view(-1,15) #(-1,5)
#    print(ump(loc_r,out_r_c,out_r_l,24,0.75).size())
#    print('r',rms.size(0))
    return rms

def testo(img,rms,NET_o):
    if rms.size()==torch.Size([0,15]):
#        print('o','0')
        return torch.tensor([]).view(-1,15)
    inp_o,loc_o=build_n(img,48,rms[:,:5])
    loc_o=loc_o.to(device)
    inp_o=inp_o.to(device)
    out_o_c,out_o_l=NET_o(inp_o)
    oms=ut.nms(ump(loc_o,out_o_c,out_o_l,48,0.8),15,0.3,True).view(-1,15) #(-1,5)
#    print('o',oms.size(0))
    return oms
    
def test_all(img):
    
    NET_p.eval()
    NET_r.eval()
    NET_o.eval()
    
    cms=testp(img,NET_p).int().float()
#    cms=c_s(cms)
#    img_c=img.copy()
#    img_draw = draw.ImageDraw(img_c)
#    cms_c=cms.cpu().detach().numpy().tolist()
#    for j in cms_c:
#        img_draw.rectangle(j[1:],fill=None,outline="red")
#        img_c.save('{}/{}.png'.format(path_out,'1'+str(i)))
    rms=testr(img,cms,NET_r)
#    rms=c_s(rms)
#    rms_c=rms.cpu().detach().numpy().tolist()
#    for j in rms_c:
#        img_draw.rectangle(j[1:5],fill=None,outline="green")
#        img_draw.ellipse((j[5]-1,j[6]-1,j[5]+1,j[6]+1),fill='green')
#        img_draw.ellipse((j[7]-1,j[8]-1,j[7]+1,j[8]+1),fill='green')
#        img_draw.ellipse((j[9]-1,j[10]-1,j[9]+1,j[10]+1),fill='green')
#        img_draw.ellipse((j[11]-1,j[12]-1,j[11]+1,j[12]+1),fill='green')
#        img_draw.ellipse((j[13]-1,j[14]-1,j[13]+1,j[14]+1),fill='green')
#    if rms.size()!=torch.Size([0]):
#        img_c.save('{}/{}.png'.format(path_out,'2'+str(i)))
#
    oms=testo(img,rms,NET_o)   
    oms=c_s(oms)
#    oms_c=oms.cpu().detach().numpy().tolist()
#    for j in oms_c:
#        img_draw.rectangle(j[1:5],fill=None,outline="blue")
#        img_draw.ellipse((j[5]-1,j[6]-1,j[5]+1,j[6]+1),fill='blue')
#        img_draw.ellipse((j[7]-1,j[8]-1,j[7]+1,j[8]+1),fill='blue')
#        img_draw.ellipse((j[9]-1,j[10]-1,j[9]+1,j[10]+1),fill='blue')
#        img_draw.ellipse((j[11]-1,j[12]-1,j[11]+1,j[12]+1),fill='blue')
#        img_draw.ellipse((j[13]-1,j[14]-1,j[13]+1,j[14]+1),fill='blue')
#    if oms.size()!=torch.Size([0]):
#        img_c.save('{}/{}.png'.format(path_out,'3'+str(i)))
#
    return oms
    
    
    

#s_time=time.time()
#torch.cuda.empty_cache()
#for i,lab in enumerate(os.listdir(path)):
##    try:
#    img=pimg.open('{}/{}'.format(path,lab))
#    img=img.convert("RGB")
#
#    oms=test_all(img,lab)    
#    print(lab)
#    torch.cuda.empty_cache()
##    except:
##        continue
#f_time=time.time()
#print(f_time-s_time)
    
#    in(img,(-1,5)) inp_r(-1,24,24,3) loc_r(-1,3)
    
#    oms_c=oms.cpu().detach().numpy().tolist()
#    for j in oms_c:
#        img_draw = draw.ImageDraw(img)
#        img_draw.rectangle(j[1:],fill=None,outline="red")
#    img.save('{}/{}.png'.format(path_out,str(i)+'1'))

#img=pimg.open('./test_img/170sss.jpg')
#img=img.convert("RGB")
#oms=test_all(img,'1170.jpg')
    
    
        
        
        
        
    



















