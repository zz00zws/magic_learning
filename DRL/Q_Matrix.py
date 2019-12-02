import torch
q = torch.zeros([6, 6]).float()
r = torch.tensor([[-1, -1, -1, -1,  0,  -1], 
               [-1, -1, -1,  0, -1, 100], 
               [-1, -1, -1,  0, -1,  -1], 
               [-1,  0,  0, -1,  0,  -1], 
               [ 0, -1, -1,  0, -1, 100], 
               [-1,  0, -1, -1,  0, 100]]).float()
alph = 0.8
e= 0.1
for time in range(101):
    state = torch.randint(0,6,[1])
    while (state != 5): 
        possible_actions = []
        possible_q =torch.tensor([0]).float().view(-1)
        for action in range(6):
            if r[state, action] >= 0:
                possible_actions.append(action)
#                print(possible_q.size(),q[state, action].size())
                torch.cat((possible_q,q[state, action].view(-1)),0)
        action = -1
        if torch.randn(1) < e:
            action = possible_actions[torch.randint(0, len(possible_actions),[1]).item()]
        else:
            action = possible_actions[torch.argmax(possible_q)]
        q[state, action] = r[state, action] + alph * q[action].max()
        state = action
    if time % 10 == 0:
        print("epoch: " ,time)
        print(q)
