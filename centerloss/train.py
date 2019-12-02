import torch
import torch.utils.data as data
from nets import net1
import torch.optim as optim
from torchvision import transforms,datasets
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

train_dataset = datasets.MNIST( root='./', train=True, transform=tf, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=1000, shuffle=True)

#net = net1().to(device)
net=torch.load('./net1.pth')
opt = optim.Adam(net.parameters())

for epoch in range(10000):
    net.train()
    for i, (l, label) in enumerate(train_loader):
        l=l.to(device)
        label=label.to(device)
        xy,z=net(l)
        x=xy[:,0].view(-1)
        y=xy[:,1].view(-1)
        mask_0=torch.lt(label,1)
        lab_0=torch.masked_select(label,mask_0)
        x_0=torch.masked_select(x,mask_0)
        y_0=torch.masked_select(y,mask_0)
        mask_1=torch.lt(label,2)
        lab_1=torch.masked_select(label,mask_1-mask_0)
        x_1=torch.masked_select(x,mask_1-mask_0)
        y_1=torch.masked_select(y,mask_1-mask_0)
        mask_2=torch.lt(label,3)
        lab_2=torch.masked_select(label,mask_2-mask_1)
        x_2=torch.masked_select(x,mask_2-mask_1)
        y_2=torch.masked_select(y,mask_2-mask_1)
        mask_3=torch.lt(label,4)
        lab_3=torch.masked_select(label,mask_3-mask_2)
        x_3=torch.masked_select(x,mask_3-mask_2)
        y_3=torch.masked_select(y,mask_3-mask_2)
        mask_4=torch.lt(label,5)
        lab_4=torch.masked_select(label,mask_4-mask_3)
        x_4=torch.masked_select(x,mask_4-mask_3)
        y_4=torch.masked_select(y,mask_4-mask_3)
        mask_5=torch.lt(label,6)
        lab_5=torch.masked_select(label,mask_5-mask_4)
        x_5=torch.masked_select(x,mask_5-mask_4)
        y_5=torch.masked_select(y,mask_5-mask_4)
        mask_6=torch.lt(label,7)
        lab_6=torch.masked_select(label,mask_6-mask_5)
        x_6=torch.masked_select(x,mask_6-mask_5)
        y_6=torch.masked_select(y,mask_6-mask_5)
        mask_7=torch.lt(label,8)
        lab_7=torch.masked_select(label,mask_7-mask_6)
        x_7=torch.masked_select(x,mask_7-mask_6)
        y_7=torch.masked_select(y,mask_7-mask_6)
        mask_8=torch.lt(label,9)
        lab_8=torch.masked_select(label,mask_8-mask_7)
        x_8=torch.masked_select(x,mask_8-mask_7)
        y_8=torch.masked_select(y,mask_8-mask_7)
        mask_9=torch.lt(label,10)
        lab_9=torch.masked_select(label,mask_9-mask_8)
        x_9=torch.masked_select(x,mask_9-mask_8)
        y_9=torch.masked_select(y,mask_9-mask_8)
        plt.scatter(x_0.cpu().detach(),y_0.cpu().detach(),color='#FF0000',s=3)
        plt.scatter(x_1.cpu().detach(),y_1.cpu().detach(),color='#AA4400',s=3)
        plt.scatter(x_2.cpu().detach(),y_2.cpu().detach(),color='#996600',s=3)
        plt.scatter(x_3.cpu().detach(),y_3.cpu().detach(),color='#669911',s=3)
        plt.scatter(x_4.cpu().detach(),y_4.cpu().detach(),color='#00FF00',s=3)
        plt.scatter(x_5.cpu().detach(),y_5.cpu().detach(),color='#009966',s=3)
        plt.scatter(x_6.cpu().detach(),y_6.cpu().detach(),color='#0066aa',s=3)
        plt.scatter(x_7.cpu().detach(),y_7.cpu().detach(),color='#2244cc',s=3)
        plt.scatter(x_8.cpu().detach(),y_8.cpu().detach(),color='#0000FF',s=3)
        plt.scatter(x_9.cpu().detach(),y_9.cpu().detach(),color='#880088',s=3)
#        plt.show()
#        label=torch.zeros(label.size()[0],10).scatter_(1,label.cpu().reshape(-1,1),1).to(device)
        loss1,loss2=net.get_loss(xy,z,label)
        loss=loss1+loss2/1000
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%10 == 0:
            print(loss.item())
    plt.show()
    torch.save(net,'./net1.pth')




