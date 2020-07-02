import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mish(nn.Module):#act
    def __init__(self):
        super().__init__()
        self.sp=nn.Softplus()

    def forward(self, x):
        return x * torch.tanh(self.sp(x))

class conv1(nn.Module):#conv+bn+mish
    def __init__(self,x,y,z,s=1,p=1,g=1,d=1,b=True):
        super().__init__()
        self.block1=nn.Sequential(  
            nn.Conv2d(x,y,z,padding=p,stride=s,
                      groups=g,dilation=d,bias=b),
            nn.BatchNorm2d(y),
            mish()
            )
    def forward(self,y):
        return self.block1(y)

class conv2(nn.Module):#conv+bn+prelu
    def __init__(self,x,y,z,s=1,p=1,g=1,d=1,b=True):
        super().__init__()
        self.block1=nn.Sequential(  
            nn.Conv2d(x,y,z,padding=p,stride=s,
                      groups=g,dilation=d,bias=b),
            nn.BatchNorm2d(y),
            nn.PReLU()
            )
    def forward(self,y):
        return self.block1(y)

class up(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(inc, outc, 3, 2, 1, 1),
            nn.BatchNorm2d(outc),
            nn.PReLU()
        )

    def forward(self, y):
        return self.block1(y)


class down(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.block1 = nn.Sequential(
            conv2(inc, outc, 3, 2, 1)
        )

    def forward(self, y):
        return self.block1(y)


class minres(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.block1=nn.Sequential(           
            conv2(x,int(x/2),1,1,0),
            conv2(int(x/2),x,3,1),
            )
    def forward(self,y):
        return self.block1(y)+y


class resblock(nn.Module):
    def __init__(self, inc, ouc, num_blocks):
        super().__init__()

        self.downsample_conv = down(inc, ouc)
        self.split_conv0 = conv2(ouc, ouc//2, 1,1,0)
        self.split_conv1 = conv2(ouc, ouc//2, 1,1,0)
        self.blocks_conv = nn.Sequential(
            *[minres(ouc//2) for _ in range(num_blocks)],
            conv2(ouc//2, ouc//2, 1,1,0)
            )
        self.concat_conv = conv2(ouc, ouc, 1,1,0)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class firstblock(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        self.downsample_conv = down(inc, ouc)
        self.split_conv0 = conv2(ouc, ouc//2, 1,1,0)
        self.split_conv1 = conv2(ouc, ouc//2, 1,1,0)
        self.blocks_conv = nn.Sequential(
            minres(ouc//2),
            conv2(ouc//2, ouc//2, 1,1,0)
            )
        self.concat_conv = conv2(ouc, ouc, 1,1,0)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class cspdarknet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1=nn.Sequential(           
            conv1(3,32,3),
            firstblock(32,64)
            )
        self.block2=resblock(64,128,2)
        self.block3=resblock(128,256,8)
        self.block4=resblock(256,512,8)
        self.block5=resblock(512,1024,4)
        
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        y=self.block4(x)
        z=self.block5(y)
        return x,y,z


class ssp(nn.Module):
    def __init__(self):
        super().__init__()
        pool_sizes=[5, 9, 13]
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features


class convx3(nn.Module):
    def __init__(self, outc, inc):
        super().__init__()
        self.block1 = nn.Sequential(
            conv2(inc,outc[0],1,1,0),
            conv2(outc[0],outc[1],3),
            conv2(outc[1],outc[0],1,1,0)
        )

    def forward(self, y):
        return self.block1(y)


class convx5(nn.Module):
    def __init__(self, outc, inc):
        super().__init__()
        self.block1 = nn.Sequential(
            conv2(inc,outc[0],1,1,0),
            conv2(outc[0],outc[1],3),
            conv2(outc[1],outc[0],1,1,0),
            conv2(outc[0],outc[1],3),
            conv2(outc[1],outc[0],1,1,0)
        )

    def forward(self, y):
        return self.block1(y)


class outhead(nn.Module):
    def __init__(self, outc, inc):
        super().__init__()
        self.block1 = nn.Sequential(
            conv2(inc,outc[0],3),
            nn.Conv2d(outc[0],outc[1],1)
        )

    def forward(self, y):
        return self.block1(y)


def out_act(out, num_anchors):
    n,c,h,w=out.size()
    out=out.permute(0,2,3,1).reshape(n,h,w,num_anchors,-1)
    out21=torch.nn.functional.sigmoid(out[:,:,:,:,:3])
    out22=torch.relu(out[:,:,:,:,3:5])
    out23=out[:,:,:,:,5:]
    return torch.cat((out21,out22,out23),4)


class mainet(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.num=num_anchors
        self.backbone = cspdarknet()

        self.conv1 = convx3([512,1024],1024)
        self.SPP = ssp()
        self.conv2 = convx3([512,1024],2048)

        self.upsample1 = up(512,256)
        self.conv_for_P4 = conv2(512,256,1,1,0)
        self.make_five_conv1 = convx5([256, 512],512)

        self.upsample2 = up(256,128)
        self.conv_for_P3 = conv2(256,128,1,1,0)
        self.make_five_conv2 = convx5([128, 256],256)
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = outhead([256, final_out_filter2],128)

        self.down_sample1 = down(128,256)
        self.make_five_conv3 = convx5([256, 512],512)

        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = outhead([512, final_out_filter1],256)

        self.down_sample2 = down(256,512)
        self.make_five_conv4 = convx5([512, 1024],1024)

        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = outhead([1024, final_out_filter0],512)


    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out_act(out0,self.num),out_act(out1,self.num),out_act(out2,self.num)



if __name__ == '__main__':
    trunk = mainet(5,11)

    x = torch.Tensor(2, 3, 608, 608)

    a,s,y_52 = trunk(x)

    print(a.shape,s.shape,y_52.shape)

