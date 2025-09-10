import os
from scipy.io import loadmat,savemat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
lis=os.listdir('data0/0HP')

N=400;Len=2048

data=np.zeros((0,Len))
label=[]
for n,i in enumerate(lis):
    path='./data0/0HP/'+i
    print('第',n,'类的数据是',path,'这个文件')
    file=loadmat(path)
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:
            files= file[key].ravel()
    data_=[]
    for i in range(N):
        start=np.random.randint(0,len(files)-Len)
        end=start+Len
        data_.append(files[start:end])
        label.append(n)
    data_=np.array(data_)

    data=np.vstack([data,data_])
label=np.array(label).reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
label = encoder.fit_transform(label)
# 7：3划分数据集

train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.3,random_state=100)

np.save('data00/0HP/train_data.npy', train_data)
np.save('data00/0HP/train_label.npy', train_label)
np.save('data00/0HP/test_data.npy', test_data)
np.save('data00/0HP/test_label.npy', test_label)
print("Train data shape:", train_data.shape)
print("Train label shape:", train_label.shape)
print("Test data shape:", test_data.shape)
print("Test label shape:", test_label.shape)