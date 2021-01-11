# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import math
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self,in_channels, f, kernel):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.f = f
        self.kernel = kernel
        self.pool = 2
        self.dilation = 1
        self.padding = int((self.kernel-1)/2)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = self.f, kernel_size = self.kernel)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.f)
        self.maxpool = nn.MaxPool2d(kernel_size = self.pool)
    
    def forward(self,x):
        x = self.conv(x)
        x = F.pad(x,(self.padding,self.padding,self.padding,self.padding))
        x = self.act(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x
    
    def getOutputDim(self, h_in, w_in):
        h_out = int((h_in + 2*self.padding - self.dilation*(self.kernel-1)-1)+1)
        w_out = int((w_in + 2*self.padding - self.dilation*(self.kernel-1)-1)+1)
        
        h_out = int(((h_out -self.dilation*(self.pool-1)-1)/self.pool) +1)
        w_out = int(((w_out -self.dilation*(self.pool-1)-1)/self.pool) +1)
        
        return h_out, w_out
        


class CNN(nn.Module):
    def __init__(self, filters, kernel, height, width):
        super(CNN, self).__init__()
        self.filters = filters
        self.kernel = kernel
        self.height = height
        self.width = width
        
        self.model = nn.Sequential()
        in_channels = 1
        h_in, w_in = height, width
        for i,f in enumerate(self.filters):
            c = ConvBlock(in_channels, f, self.kernel)
            self.model.add_module("conv_block_{}".format(i+1), c)
            in_channels = f
            h_in, w_in = c.getOutputDim(h_in, w_in)
            
        self.model.add_module("flatten_1",nn.Flatten())
        self.model.add_module("linear_1", nn.Linear(f*h_in*w_in, 16))
        self.model.add_module("relu_out1",nn.ReLU())
        self.model.add_module("batch_norm1d",nn.BatchNorm1d(16))
        self.model.add_module("dropout_1",nn.Dropout(0.5))
        self.model.add_module("linear_2",nn.Linear(16,4))
        self.model.add_module("relu_out2",nn.ReLU())
        self.model.add_module("linear_3",nn.Linear(4,1))
        
        
        '''
        self.model = nn.Sequential()
        in_channels = 1
        for i,f in enumerate(self.filters):
            self.model.add_module("conv_2d_{}".format(i+1),nn.Conv2d(in_channels,f,self.kernel))
            self.model.add_module("relu_{}".format(i+1),nn.ReLU())
            self.model.add_module("batch_norm2d_{}".format(i+1),nn.BatchNorm2d(f))
            self.model.add_module("max_pool_2d_{}".format(i+1),nn.MaxPool2d(kernel_size = 2))
            in_channels = f
        
        self.model.add_module("flatten",nn.Flatten())
        self.model.add_module("linear_1", nn.Linear(f, 16))
        self.model.add_module("relu_out1",nn.ReLU())
        self.model.add_module("batch_norm1d",nn.BatchNorm1d(16))
        self.model.add_module("dropout_1",nn.Dropout(0.5))
        self.model.add_module("linear_2",nn.Linear(16,4))
        self.model.add_module("relu_out2",nn.ReLU())
        self.model.add_module("linear_3",nn.Linear(4,1))
        '''
   

    def forward(self,x):
        '''
        for i,f in enumerate(self.filters):
            x = nn.Conv2d(x.size()[-1],f,self.kernel)(x)
            x = nn.ReLU()(x)
            x = nn.BatchNorm2d(f)(x)
            x = F.max_pool2d(x,(2,2))

        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = nn.Linear(x.size(), 16)(x)
        x = nn.ReLU()(x)
        x = nn.BatchNorm1d(x.size()[-1])(x)
        x = nn.Dropout(0.5)(x)
        x = nn.Linear(16,4)(x)
        x = nn.ReLU()(x)
        x = nn.Linear(4,1)(x)
        return x
        '''
        return self.model(x)

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
    
    def forward(self, output, target):
        return self.dtw(output, target, 1)

    
class SpatialTemporalLoss(nn.Module):
    def __init__( self):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

     
    def pearsonCorrelation(self,output, target):
        vx = output - torch.mean(output)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost


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

    def forward(self,output, target):
        #output, target = Variable(output, requires_grad = True).to(self.device), Variable(target, requires_grad = True).to(self.device)
        spatialTerm = nn.MSELoss()(output,target)
        temporalTerm = self.dtw(output, target, 1)
        correlationTerm = self.pearsonCorrelation(output,target)
        #return torch.mul(torch.sqrt(torch.add(spatialTerm ** 2, temporalTerm ** 2)), (1-correlationTerm))
        #return torch.mul(spatialTerm,torch.mul(temporalTerm,(1-correlationTerm)))
        #return torch.log(1 + spatialTerm * temporalTerm)
        




def run(train_X,valid_X, test_X, train_Y, valid_Y,test_Y, epochs, batch_size, filters,kernel, lr,best_loss, gpu):
    device = 'cuda:0' if gpu else 'cpu'
    train_X, valid_X, test_X = torch.from_numpy(train_X).to(device),torch.from_numpy(valid_X).to(device),torch.from_numpy(test_X).to(device)
    train_Y, valid_Y, test_Y = torch.from_numpy(train_Y).to(device),torch.from_numpy(valid_Y).to(device),torch.from_numpy(test_Y).to(device)

    train_X, valid_X, test_X = train_X.permute(0,3,1,2), valid_X.permute(0,3,1,2), test_X.permute(0,3,1,2)
    train_X, valid_X, test_X = train_X.float(), valid_X.float(), test_X.float()
    train_Y, valid_Y, test_Y = train_Y.float(), valid_Y.float(), test_Y.float()
    

    model = CNN(filters, kernel,train_X.size()[2], train_X.size()[3])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=int(epochs/10), 
                                                      threshold=0.0001, cooldown=0, min_lr=0.0000001, verbose=True)
    
    loss_fn1 = nn.MSELoss()
    loss_fn2 = PearsonLoss()
    loss_fn3 = TemporalLoss()
    a, b, c = 1, 0, 0
    
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        model.train()
        for i in range(0, train_X.size()[0], batch_size):
            optimizer.zero_grad()
            batch_x, batch_y = train_X[i:i+batch_size], train_Y[i:i+batch_size]
            
            outputs = model.forward(batch_x)
            batch_y = torch.unsqueeze(batch_y,1)
            
            #loss = a*loss_fn1(outputs,batch_y) - b*(1-loss_fn2(outputs, batch_y)) + c*loss_fn3(outputs, batch_y) 
            loss = loss_fn1(outputs,batch_y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        
        model.eval()
        val_outputs = model.forward(valid_X)
        mse_val, corr_val, dtw_val = loss_fn1(val_outputs,valid_Y), loss_fn2(val_outputs, valid_Y), loss_fn3(val_outputs, valid_Y) 
        #val_loss = a*mse_val - b*(1-corr_val) + c*dtw_val
        val_loss = mse_val
        #train_outputs = model.forward(train_X)
        #val_loss = loss_fn(train_outputs, train_Y)
        #mse, corr, dtw = torch.mean((train_outputs-train_Y)**2), loss_fn.pearsonCorrelation(train_outputs, train_Y), loss_fn.dtw(train_outputs, train_Y,1)


        scheduler.step(val_loss)
        end = time.time()
        print("Epochs {}/{}: {}s  train_loss: {:.4e}  val_loss: {:.4e}".format(epoch+1,epochs,round(end-start,1),train_loss/(math.ceil(train_X.size()[0]/batch_size)),val_loss.item()))
        print("\t\t\tValidation\tMSE: {:.4e}, CORR: {:.4e}, DTW: {}".format(mse_val, corr_val, dtw_val))
        
    model.eval()
    test_preds = model.forward(test_X)
    mse, corr, dtw = loss_fn1(test_preds,test_Y), loss_fn2(test_preds, test_Y), loss_fn3(test_preds, test_Y) 
    test_loss = a*mse - b*(1-corr) + c*dtw

    if test_loss < best_loss:
        torch.save(model, 'CNNtorch')

        with open("BestCNNtorch.txt","w+") as bestFile:
            bestFile.write("Epochs: {}\nBatch Size: {}\nLearning Rate: {}\nFilters:{}\nKernel Size:{}".format(epochs, batch_size,lr,filters,kernel))

    return test_preds, test_loss, mse, corr, dtw