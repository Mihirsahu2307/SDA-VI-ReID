from __future__ import division, print_function, absolute_import

import torch
import torch.nn as nn

from functools import partial
from torch.autograd import Variable

from torch.nn import functional as F


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


class DMMD(nn.Module):

    """
    Implementation of MMD :
    https://github.com/shafiqulislamsumon/HARTransferLearning/blob/master/maximum_mean_discrepancy.py
    """
    
    # We have modified the implementation of DMMD slightly

    def __init__(self, use_gpu=True, P=4, K=4, hdmmd=False):
        super(DMMD, self).__init__()
        self.use_gpu = use_gpu
        # self.batch_size = batch_size
        # self.instances = instances
        self.P = P
        self.K = K
        self.hdmmd = hdmmd

    def mmd_linear(self, f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)

    def mmd_rbf_accelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i+1)%batch_size
            t1, t2 = s1+batch_size, s2+batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return loss / float(batch_size)

    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

    def pairwise_distance(self, x, y):

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)
        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_.cuda())
        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))
        return cost

    def mmd_loss(self, source, target):

        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
            )
        loss_value = self.maximum_mean_discrepancy(source, target, kernel=gaussian_kernel)
        loss_value = loss_value
        return loss_value

    def forward(self, source_features, target_features):
        
        # if hdmmd:
        #     s, t = list(torch.split(source_features,self.K,dim=0)), list(torch.split(target_features,self.K,dim=0))
        # else:
        s, t = list(torch.split(source_features,self.K,dim=0)), list(torch.split(target_features,self.K,dim=0))

        wct = torch.tensor([]).to(torch.device('cuda'))
        bct = torch.tensor([]).to(torch.device('cuda')) 
        wcs = torch.tensor([]).to(torch.device('cuda'))
        bcs = torch.tensor([]).to(torch.device('cuda'))
        
        for i in range(len(t)):
            mat1 = t[i]
            mat1.cuda()
            wct = torch.cat((wct, compute_distance_matrix(mat1, mat1)))
            for j in range(len(t)):
                if i == j: continue
                
                mat2 = t[j]
                mat2.cuda()
                
                bct = torch.cat((bct, compute_distance_matrix(mat1, mat2)))
                    
        for i in range(len(s)):
            mat1 = s[i]
            mat1.cuda()
            
            wcs = torch.cat((wcs, compute_distance_matrix(mat1, mat1)))
            for j in range(len(s)):
                if i == j: continue
                
                mat2 = s[j]
                mat2.cuda()
                
                bcs = torch.cat((bcs, compute_distance_matrix(mat1, mat2)))
            
        # We want to modify only target distribution
        bcs = bcs.detach()
        wcs = wcs.detach()

        if self.hdmmd:
            return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct)
        else:
            return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct), self.mmd_loss(source_features, target_features)
    
    
class HDMMD(nn.Module):

    def __init__(self, P=4, K=4):
        super(HDMMD, self).__init__()
        self.P = P
        self.K = K
        self.criterion_dmmd = DMMD(hdmmd=True, P=P, K=K).to(torch.device('cuda'))
        
    # def forward(self, source_features, target_features):
    def forward(self, source_rgb_features, source_ir_features, target_rgb_features, target_ir_features):
        
        loss_rgb = sum(self.criterion_dmmd(source_rgb_features, target_rgb_features)) / 2
        loss_ir = sum(self.criterion_dmmd(source_ir_features, target_ir_features)) / 2
        
        return (loss_rgb + loss_ir)
        