import trainer
import nets

if __name__=='__main__':
    net=nets.pnet
    tr=trainer.trains(net,'./net_p.pth','./data/img_train/12')
    tr.train()




















