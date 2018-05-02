import os
import numpy as np
from numpy import *
inputfile='watermelon.txt'
all_data=list()
with open(inputfile,'r') as f:
    data=f.readlines()
    #print(data)
    for line in data[1:]:
        #print(line)
        odom=line.split()
        data_float=list(map(float,odom[1:]))
        all_data.append(data_float)
print(all_data)
x=np.array(all_data)[:,:-1]
#ones=np.ones(len(x))
x=np.insert(x,len(x[0]),1,axis=1).T#take the b into consideration
y=np.array(all_data)[:,-1]
# print("x:",x.T)
# print("y:",y)
# print(x.shape)
# print("x.T:",x.T)
# xt=x[0].reshape(3,1)
# print("x[0].T:",xt)
#set the hypher para
sample_size=x.shape[1]
para_size=x.shape[0]
beta=np.zeros((para_size,1))
#print(beta)
old_l=0
steps=0
run=1
#set loss fuction and iter
while run:
#for _ in range(4):
    acc_l=0
    grad1=np.zeros((para_size,1))
    grad2=np.zeros((para_size,para_size))
    beta_x = np.dot(beta.T, x)[0]
    for i in range(sample_size):
        xi=x[:,i].reshape(para_size,1)
        xit=x[:,i].reshape(1,para_size)
        p=np.exp(beta_x[i])/(np.exp(beta_x[i])+1)
        acc_l=acc_l+(-y[i]*beta_x[i]+np.log(1+np.exp(beta_x[i])))
        grad1=grad1-xi*(y[i]-p)
        grad2=grad2+np.dot(xi,xit)*p*(1-p)
    if np.abs(acc_l-old_l)<=0.0001:
        print("finish after {} steps".format(steps))
        print("last_beta: \n", beta)
        run=0
    steps+=1
    old_l=acc_l
    beta=beta-np.dot(np.linalg.inv(grad2),grad1)
#calculate the accuracy:
t=0
for i in range(sample_size):
    z=np.dot(beta.T,x[:,i].reshape(para_size,1))
    if (z>0 and y[i]==1) or (z<=0 and y[i]==0):
        t+=1
print("accuracy:{}".format(t*100/sample_size))









