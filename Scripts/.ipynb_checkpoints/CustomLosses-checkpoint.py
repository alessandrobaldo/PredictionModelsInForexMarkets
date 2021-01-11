import torch.nn as nn
import torch

'''PYTORCH LOSSES'''
class PearsonLoss(nn.Module):
    def __init__( self):
        super(PearsonLoss, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def pearsonCorrelation(self,output, target):
        vx = output - torch.mean(output)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

    def forward(self, output, target):
        return self.pearsonCorrelation(output, target)

class TemporalLoss(nn.Module):
    def __init__( self):
        super(TemporalLoss, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def dtw(self,output, target, window):
        n, m = output.size()[0], target.size()[0]
        w = torch.max(torch.tensor([window, abs(n-m)]).to(self.device))
        dtw_matrix = torch.zeros((n+1, m+1), requires_grad = True).to(self.device)
        dtw_matrix = torch.add(dtw_matrix, float("Inf"))
        dtw_matrix[0, 0] = 0
        for i in range(1, n+1):
            #for j in range(torch.max(torch.tensor([1, i-w]).to(self.device)), torch.min(torch.tensor([m, i+w]).to(self.device))+1):
                #dtw_matrix[i,j] = 0
            a, b = torch.max(torch.tensor([1, i-w]).to(self.device)), torch.min(torch.tensor([m, i+w]).to(self.device))+1
            dtw_matrix[i,a:b] = torch.tensor([0]*(b-a)).to(self.device)
            
            for j in range(a, b):
                cost = torch.abs(output[i-1] - target[j-1])
                last_min = torch.min(torch.tensor([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]).to(self.device))
                dtw_matrix[i, j] = torch.add(cost,last_min)

        return dtw_matrix[-1,-1]


import keras.backend as K
from keras.losses import MSE

'''KERAS LOSSES'''
def MSExp(output, target):
    return K.exp(MSE(output, target)) - 1

def PearsonLoss(output, target):
    output_c = output - K.mean(output)
    target_c = target - K.mean(target)
    r_num = K.sum(output_c * target_c)
    r_den = K.sqrt(K.sum(K.square(output_c)) * K.sum(K.square(target_c)))
    r = r_num / r_den
    return 1 - r**2

def DTWLoss(output, target, window=1):
    n, m = K.int_shape(output), K.int_shape(target)
    n, m = 32, 32
    w = K.maximum(window, K.abs(n-m))
    dtw = K.ones((n+1, m+1))*1e8
    dtw[0,0] = 0
    for i in range(1, n+1):
        a, b = K.maximum(K.variable(1),i-w), K.minimum(m, i+w)+1
        dtw[i, a:b] = K.zeros((1, b-a))

        for j in range(a, b):
            cost = K.abs(output[i-1] - target[j-1])
            last_min = K.minimum(dtw[i-1,j], dtw[i, j-1])
            last_min = K.minimum(last_min, dtw[i-1, j-1])

    return dtw[-1, -1]

def AdaptiveMSE(output, target):
    thresh = 5e-4
    mse = MSE(output, target)
    msqe = K.mean(K.sqrt(K.abs(output - target)))
    mask = K.greater(mse,thresh)
    
    return K.switch(mask, mse, msqe)


	
	