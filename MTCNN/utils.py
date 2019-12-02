import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def iou(x11,xs):
    x11=x11.to(device)
    xs=xs.to(device)
    a_x1=(x11[:,3]-x11[:,1])*(x11[:,4]-x11[:,2])
    a_x=(xs[:,3]-xs[:,1])*(xs[:,4]-xs[:,2])
    x1=torch.max(x11[:,1],xs[:,1])
    y1=torch.max(x11[:,2],xs[:,2])
    x2=torch.min(x11[:,3],xs[:,3])
    y2=torch.min(x11[:,4],xs[:,4])
    
    w=torch.max(torch.tensor([0]).float().to(device),x2-x1).to(device)
    h=torch.max(torch.tensor([0]).float().to(device),y2-y1).to(device)
    
    s=w*h/(a_x+a_x1-w*h)
    return s

def iou_m(x11,xs):
    x11=x11.to(device)
    xs=xs.to(device)
    a_x1=(x11[:,3]-x11[:,1])*(x11[:,4]-x11[:,2])
    a_x=(xs[:,3]-xs[:,1])*(xs[:,4]-xs[:,2])
    x1=torch.max(x11[:,1],xs[:,1])
    y1=torch.max(x11[:,2],xs[:,2])
    x2=torch.min(x11[:,3],xs[:,3])
    y2=torch.min(x11[:,4],xs[:,4])
    
    w=torch.max(torch.tensor([0]).to(device).float(),x2-x1).to(device)
    h=torch.max(torch.tensor([0]).to(device).float(),y2-y1).to(device)
    
    s=w*h/torch.min(a_x,a_x1)
    return s

def nms(boxes,size,thresh=0.3,isMin=False):
    if boxes.shape[0] == 0:
        return torch.tensor([])
    asd,boxx=(-boxes[:,0]).sort(0)
    _boxes = boxes[boxx].to(device)
    r_boxes = torch.tensor([]).view(-1,size).to(device)
    while _boxes.shape[0] >1:
        a = _boxes[0].view(-1,size)
        b = _boxes[1:].view(-1,size)
        r_boxes = torch.cat((r_boxes,a),0)
        if isMin:
            _boxes = b[iou_m(a[:,:5],b[:,:5]) < thresh]
        else:
            _boxes = b[iou(a[:,:5],b[:,:5]) < thresh]
    if _boxes.shape[0] >0:
        r_boxes = torch.cat((r_boxes,_boxes[0].view(-1,size)),0)
    return r_boxes

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    