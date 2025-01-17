import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import sys
from tqdm import tqdm

print(torch.version.cuda) #10.1
t3 = time.time()
##############################################################################
"""args for AE"""

args = {}
args['img_size'] = 256
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 1#3    # number of channels in the input data

args['n_z'] = 300 #600     # number of dimensions in latent space.

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 10         # how many epochs to run for (original used 200)
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'mnist'  #'fmnist' # specify which dataset to use
args['num_class'] = 2

##############################################################################



## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1), # output dimension: 128 X 128 X 16
           # Each Kernal Size : 4 X 4 X 3
           """ 16 Kernel used in the 1st Convolutional layer so the output will be 128 X 128 X 16 """
           nn.LeakyReLU(0.2, inplace=True),
            
           # Each Kernal Size : 4 X 4 X 16
           nn.Conv2d(16, 128, kernel_size=4, stride=2, padding=1), # 64 X 64 X 128
           nn.LeakyReLU(0.2, inplace=True),
            
           # Each Kernal Size : 4 X 4 X 128
           nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32 X 32 X 256
           nn.ReLU(),
        
           nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Output : 16 X 16 X 512
           # Each Kernal Size : 4 X 4 X 256 
           """ 4 X 4 X 256 = 4096 .When you use a kernel size of 4x4x256 in a convolutional 
           neural network, what happens is that for each location in the input volume, 
           there will be 4096 point-wise multiplications between the corresponding 
           values in the input volume and the kernel. These multiplications are then 
           added together to produce a single scalar value. This process occurs for 
           every location in the input volume, effectively transforming it into an output 
           volume with reduced spatial dimensions (16 X 16) """
           nn.LeakyReLU(0.2, inplace=True)
        )

        # Linear layers to project to latent space
        self.fc = nn.Linear(512*16*16,)  # Output: latent space (n_z)


    def forward(self, x, labsn):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = self.conv(x)
        
        # Class images Bagging
        num_class_in_sample = np.unique(labsn)
        train_list = [[] for _ in range(args['num_class'])]
        list_class_latent = [[] for _ in range(args['num_class'])]

        for i, label in enumerate(labsn):
          train_list[label].append(torch.tensor(x[i].cpu().detach().numpy()))

        for i, label in enumerate(labsn):
          test = torch.stack(train_list[label])
          #Each class
          x0 =  test.squeeze().to(device) # Move x0 to the same device as self.fc
          x0 = x0.view(x0.size(0), -1)
          x0_linear = self.fc(x0)
          list_class_latent[label] = x0_linear

        # Mixed Class

       
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        #sys.exit(1)

        return x, list_class_latent


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 16, self.dim_h * 8, 4, 2, 1, bias=False), # Adjusted layer for 256x256
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            #nn.Sigmoid())
            nn.Tanh())

    def forward(self, x):
        #print('dec')
        #print('input ',x.size())
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x

##############################################################################
"""set models, loss functions"""
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


##############################################################################
"""functions to create SMOTE images"""

def biased_get_class(c):

    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]

    return xbeg, ybeg
    #return xclass, yclass


def G_SM(X, y,n_to_sample,cl):

    # determining the number of samples to generate
    #n_to_sample = 10

    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

#xsamp, ysamp = SM(xclass,yclass)
###############################################################################
def calculate_diffs(z):
    """
    Calculate the differences between elements in the input list 'z'
    and return the total sum of all differences for each original element.

    Args:
        z (list): A list of lists containing numerical values.

    Returns:
        list: A list containing the total sum of differences for each original element in 'z'.
    """
    all_diffs_list = []  # List to store lists of differences
    for i, arr1 in enumerate(z):
        diffs_for_arr1 = []  # Store differences for this arr1
        for j, arr2 in enumerate(z):
            if i != j:  # Skip comparing with itself
                diffs = []  # Store diffs for this pair of arr1, arr2
                for el1 in arr1:
                    # Calculate the sum of Euclidean distances between el1 and all elements in arr2
                    diff_for_el1 = sum(np.linalg.norm(np.array([el1.cpu().detach().numpy()]) - np.array([el2.cpu().detach().numpy()])) for el2 in arr2)
                    diffs.append(diff_for_el1)  # Append diff for el1 to diffs
                diffs_for_arr1.append(diffs)  # Append the diffs list for the current arr2
        all_diffs_list.append(diffs_for_arr1)  # Append the diffs_for_arr1

    # Sum all elements within each array to get a single total sum
    total_sums = [sum(sum(inner_list) for inner_list in outer_list) for outer_list in all_diffs_list]

    return total_sums

###############################################################################

