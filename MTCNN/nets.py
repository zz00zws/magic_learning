import torch.nn as nn       
        
class pnet(nn.Module):

    def __init__(self):
        super(pnet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1,padding=1),  # conv1
            nn.BatchNorm2d(10),
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(16),
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(32),
            nn.PReLU()  # PReLU3
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x1 = self.pre_layer(x)
        cond = self.layer1(x1)
        offset = self.layer2(x1)
        return cond,offset

class rnet(nn.Module):
    def __init__(self):
        super(rnet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1,padding=1),  # conv1
            nn.BatchNorm2d(28),
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(48),
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.BatchNorm2d(64),
            nn.PReLU()  # prelu3

        )

        self.conv4 = nn.Sequential( nn.Linear(64 * 3 * 3, 128),
        nn.BatchNorm1d(128),
        nn.PReLU() 
        )

        self.layer1 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 4)
        )

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)


        label = self.layer1(x)
        offset = self.layer2(x)
        return label, offset


class onet(nn.Module):
    def __init__(self):
        super(onet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1),  # conv1
            nn.BatchNorm2d(32),
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(64),
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(64),
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.BatchNorm2d(128),
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.PReLU() )

        self.layer1 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)


        label = self.layer1(x)
        offset = self.layer2(x)
        return label, offset
        
        
        
        
        
        
        
        
        
