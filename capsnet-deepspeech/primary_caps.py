#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

from squash import squash

class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            
           
            mask = torch.cuda.ByteTensor(x.size()).fill_(0)
           
            if x.is_cuda:
                mask = mask.cuda()
               
            
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths
    
class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels ,
                              out_channels=out_channels ,
                              kernel_size=16 ,
                              stride=4,
                              bias=True)
        
        
        #self.conv = nn.Sequential(    
        #    nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
        #    nn.BatchNorm2d(32),
        #    nn.Hardtanh(0, 20, inplace=True)
        #)
        
    def forward(self, x):
        # x: [batch_size, in_channels=256, 20, 20]

        h = self.conv(x)
        # h: [batch_size, out_channels=8, 6, 6]
        return h


class PrimaryCaps(nn.Module):
	def __init__(self):
		super(PrimaryCaps, self).__init__()

		self.conv1_out = 800 # out_channels of Conv1, a ConvLayer just before PrimaryCaps
		self.capsule_units = 50
		self.capsule_size = 16

		def create_conv_unit(unit_idx):
				unit = ConvUnit(
					in_channels=self.conv1_out, 
					out_channels=self.capsule_size
				)
				self.add_module("unit_" + str(unit_idx), unit)
				return unit

		self.conv_units = [create_conv_unit(i) for i in range(self.capsule_units)]

	def forward(self, x):
		# x: [256, 20, 20]
		
		u = []
		for i in range(self.capsule_units):
			u_i = self.conv_units[i](x)
			# u_i: [capsule_size=8, 6, 6]

			u_i = u_i.view(self.capsule_size, -1, 1)
			# u_i: [capsule_size=8, 36, 1]

			u.append(u_i)
		# u: [batch_size, capsule_size=8, 36, 1] x capsule_units=32

		u = torch.cat(u, dim=3)
		# u: [batch_size, capsule_size=8, 36, capsule_units=32]

		u = u.view(batch_size, self.capsule_size, -1)
		# u: [batch_size, capsule_size=8, 1152=36*32]

		u = u.transpose(1, 2)
		# u: [batch_size, 1152, capsule_size=8]

		u_squashed = squash(u, dim=2)
		# u_squashed: [batch_size, 1152, capsule_size=8]

		return u_squashed

		