import numpy as np
import pickle
import pandas as pd

df = pd.read_csv('/home/cxu-serve/p1/lchen63/trustyAI/train.csv')
#print(len(df['filename']),len(df['label']))

fake = df[df['label'] == 1]
real = df[df['label'] == 0]
print('Real: ', len(real['filename']), ' Fake: ', len(fake['label']))
fake_list = fake['filename'].to_list()
real_list = real['filename'].to_list()

path = 'output/pywork/' #/fconf.pckl'

t = []
for p in fake_list[:100]: 
    path1 = path+p+'/fconf.pckl'
    with open(path1, 'rb') as f:
        data = pickle.load(f)
    t.append(data)

with open('output/fake.pckl', 'wb') as fil:
        pickle.dump(t, fil)
