import torch
import torch.nn as nn
import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CBOW(nn.Module):

    def __init__(self, voc_num, voc_dim):
        super().__init__()

        self.codebook = nn.Embedding(voc_num, voc_dim)
        # self.codebook = nn.Parameter(torch.randn(voc_num,voc_dim))

        self.linear_1 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear_2 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear_4 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear_5 = nn.Linear(voc_dim, voc_dim, bias=False)

    def forward(self, x1, x2, x4, x5):
        v1 = self.codebook(x1)
        v2 = self.codebook(x2)
        v4 = self.codebook(x4)
        v5 = self.codebook(x5)
#        print(v1.size())

        y1 = self.linear_1(v1)
        y2 = self.linear_1(v2)
        y4 = self.linear_1(v4)
        y5 = self.linear_1(v5)

        return y1 + y2 + y4 + y5

    def getLoss(self, x3, y3):
        v3 = self.codebook(x3)

        return torch.mean((y3 - v3) ** 2)
    
class attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.lay1 = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)      
        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        
    def forward(self,x):
        x = self.lay1(x)
        
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)
        q, k, v = x.transpose(-2, -3).chunk(3, dim=-1)
        w = (q @ k.transpose(-1, -2)) / self.dk
        mask = torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)).to(device)
        mask = mask[0:w.size(-2), 0:w.size(-1)]
        w = torch.softmax(w * mask - (1 - mask) * 1e6, dim=-1)
        a = w @ v
        a = a.transpose(-2, -3)
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)
        h = self.c_proj(a)
        return h
    
class block(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)

        self.attention = attention()

        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4*cfg.embed_dim),
            nn.PReLU(),
            nn.Linear(4*cfg.embed_dim, cfg.embed_dim),
        )
        
    def forward(self, x):
        h = self.layer_normal_1(x)
        a = self.attention(h)
        a = a + x
        a = self.layer_normal_2(a)
        h = self.proj(a)
        y = h + a
        return y

class gpt(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)

        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(block())

        self.sequential = nn.Sequential(*self.blocks)

        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)

    def forward(self, x, p):
        e = self.vocab_embed(x)
        p = self.pos_embed(p)
        h = e+p
        h = self.sequential(h)
        return self.output_layer(h)


if __name__ == '__main__':
    net=gpt().cuda()
    x = torch.tensor([[0]]).cuda()
    p = torch.tensor([[0]]).cuda()
    for i in range(100):
        y = net(x, p)
        y = y[:, -1:]
        v, y = torch.topk(y, 8, dim=-1)

        v, y = v.reshape(-1, 8), y.reshape(-1, 8)
        v = torch.multinomial(torch.softmax(v, dim=-1), 1)
        y = torch.gather(y, -1, v)

        x = torch.cat([x, y], dim=1)
        p = torch.tensor([range(i + 2)]).cuda()

        print(x)














































