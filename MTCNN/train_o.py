import trainer
import nets

if __name__=='__main__':
    net=nets.onet
    tr=trainer.trains(net,'./net_o.pth','./data/img_train/48')
    tr.train()




















