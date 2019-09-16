import os
import numpy as np
import pickle
import math
import torch

result_dir = 'Your result numpy dir'
dicts = {}
dataset = 'cdvs_test_retrieval'
lambda_RHD = 0.5
lambda_R = 0.12
bitave = 1024
with open('../data/gnd_{}.pkl'.format(dataset),'rb') as f:
    cfg = pickle.load(f)
## load cdvs global descriptor
global_cdvs=cfg['imlist_global']
## load cnns deep descriptor
vecs = np.load(os.path.join(result_dir, "{}_vecs_resize.npy".format(dataset)))
vecs=vecs.T
def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = A*BT
    
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1])) 
    
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1)) 
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    for i in range(len(SqED)):
        for j in range(len(SqED)):
            if SqED[i,j] <= 0:
                SqED[i,j] = 0
    ED = (SqED.getA())**0.5
    return np.matrix(ED)

vecs = np.hstack((vecs, global_cdvs * lambda_RHD))
vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
x_dis = (EuclideanDistances(np.matrix(vecs), np.matrix(vecs)))
[ii, jj] = x_dis.shape
allsum = x_dis.sum(1)
result = np.zeros((len(cfg['imlist'])))
for i in cfg['gnd_id']:
    cluster=i['ok']
    if len(cluster) > 1:
        cluster_inner = np.zeros((len(cluster),len(cluster)))
        for idx1, mou1 in enumerate(cluster):
            for idx2, mou2 in enumerate(cluster):
                cluster_inner[idx1,idx2] = x_dis[mou1, mou2]
        inner = cluster_inner.sum(1)
    else:
        inner = [0]
    for idx1, mou1 in enumerate(cluster):
        inn = inner[idx1]
        if inner[idx1] != 0:
            inn=inner[idx1]/(len(cluster)-1)
        inter = (allsum[mou1]-inner[idx1])/(len(cfg['imlist'])-len(cluster))
        result[mou1] = inn / inter
result_t = torch.from_numpy(result*1000000)
result_fen, result_ming = torch.sort(result_t)
result_fen = result_fen.tolist()
result_ming = result_ming.tolist()
zero_num = [i == 0 for i in result_t].count(True)
total = len(result_ming)-zero_num

with open('../data/result_global_{}_{}.txt'.format(str(bitave), str(lambda_R)), 'w') as f:
    for idx, i in enumerate(cfg['imlist']):
        mingci = result_ming.index(idx)
        if mingci < zero_num:
            f.write(str(bitave)+'\n')
        else:
            fen=math.ceil(bitave+lambda_R*(mingci-zero_num-total/2))
            f.write(str(fen)+'\n')

