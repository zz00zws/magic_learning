import trainer
import nets

if __name__=='__main__':
    net=nets.rnet
    tr=trainer.trains(net,'./net_r.pth','./data/img_train/24')
    tr.train()




















