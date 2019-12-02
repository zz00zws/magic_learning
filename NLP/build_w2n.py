import cfg
import os
#import jieba

#with open(cfg.paper_path,encoding='UTF-8') as f:
#    a=f.read(1)
wordlist=[]
#for i in a:
#    wordlist.append(i)
#wordk=list(set(wordlist))
#worddic=dict(zip(wordk,range(len(wordk))))
#f=open(cfg.word_path,'a+',encoding='UTF-8')
#f.write('[START] [SEQ] [UNK] [PAD] ')
#for i in wordk:
#    f.write(i+' ')
#
#f1=open(cfg.num_path,'a+',encoding='UTF-8')
#
#for i in wordlist:
##    print(str(worddic[i]))
#    f1.write(str(worddic[i])+' ')
#
words = set()


for i in os.listdir(cfg.paper_path):
    print(i)
    f_path = os.path.join(cfg.paper_path,i)
    with open(f_path, "r+", encoding="UTF-8") as f:
        w = f.read(1)
        while w:
    
            if w == '\n' :
                words.add('[space]')
                wordlist.append('[space]')
            elif w == ' ' or w == '\r':
                pass
            else:
                words.add(w)
                wordlist.append(w)
            w = f.read(1)

with open(cfg.word_path, "w+", encoding="UTF-8") as f:
    f.write("[START] [SEQ] [UNK] [PAD] [END] ")
    f.write(" ".join(words))
    f.flush()

print(len(words))
worddic=dict(zip(words,range(5,len(words)+5)))
f1=open(cfg.num_path,'w+',encoding='UTF-8')

for i in wordlist:
    f1.write(str(worddic[i])+' ')
#print(wordlist)
#index=map(str,wordlist)
#f1.write(" ".join(worddic[index]))
#for i in wordlist:
#    print(str(worddic[i]))
#    f1.write(str(worddic[i])+' ')





    


























































