import torch
import torch.nn as nn
import cfg
import os
import nets
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):

    def __init__(self, dir):

        self.dataset = []

        with open(dir,"r+") as f:
            ws = [int(x) for x in f.readline().split()]
            ws_len = len(ws)
            start = 0
            while ws_len - start > cfg.pos_num + 1:
                self.dataset.append(ws[start:start + cfg.pos_num + 1])
                start += 50
            else:
                if ws_len > cfg.pos_num + 1:
                    self.dataset.append(ws[ws_len - cfg.pos_num - 1:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index],dtype =torch.long)
        return data[0:-1], data[1:]

class Trainer():

    def __init__(self):
        self.fun1 = nn.CrossEntropyLoss()
        if os.path.exists(cfg.save_path):
            self.net = torch.load(cfg.save_path).to(device)
        else:
            self.net = nets.gpt().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def train(self):
        myDataset = MyDataset(cfg.num_path)
        dataloader = DataLoader(myDataset, batch_size=13,shuffle=True,drop_last=True)
        for epoch in range(100000):
            sum_loss = 0
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                p = torch.arange(0, x.shape[1])[None, :].repeat(x.shape[0], 1).to(device)
                _y = self.net(x, p).reshape(-1, cfg.vocab_num)
                y = y.reshape(-1).long()
                loss = self.fun1(_y,y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(epoch, i, loss.cpu().detach().item())
                sum_loss += loss.cpu().detach().item()
                if i%100==0 and i>0:
                    torch.save(self.net,cfg.save_path)
            print(epoch, sum_loss / len(dataloader))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
