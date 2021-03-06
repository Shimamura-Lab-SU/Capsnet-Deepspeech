#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pytorch_memlab import profile
import torch.nn.functional as F
import pdb
import math

from squash import squash

device = torch.device("cuda")
class DigitCaps(nn.Module):
	def __init__(self):
		super(DigitCaps, self).__init__()

		self.routing_iters = 3
		#self.module = module        
		#self.gpu = gpu
		#カプセルの設定
		self.in_capsules = 50
		self.in_capsule_size = 9
		self.out_capsules = 29
		self.out_capsule_size = 8

		self.W = nn.Parameter(
			torch.Tensor(
				self.in_capsules, 
				self.out_capsules, 
				self.out_capsule_size, 
				self.in_capsule_size
			)
		)
		#カプセル化のパラメータ
		self.conv = nn.Conv2d(
			in_channels=1,
			out_channels=50,
			kernel_size=10,
			stride=5,
			bias=True
		)
		# W: [in_capsules, out_capsules, out_capsule_size, in_capsule_size] = [200, 29, 8, 4]
		self.reset_parameters()

	def reset_parameters(self):
		""" Reset W.
		"""
		stdv = 1. / math.sqrt(self.in_capsules)
		self.W.data.uniform_(-stdv, stdv)

	# FIXME, write in an easier way to understand, some tensors have some redundant dimensions.
	#@profile
	def forward(self, x):
		# x: [ batch_size,in_capsules=1152, in_capsule_size=8]
		#カプセル化(畳み込み)
		si = x.size(0)
		x = torch.reshape(x,(si,20 ,20))
		x = x.unsqueeze(1)
		x = self.conv(x)
		x = x.view(si,x.size(1),-1)
		#x = torch.reshape(x,(x.size(0),x.size(1),self.in_capsules ,self.in_capsule_size ))

		x = torch.stack([x] * self.out_capsules, dim=2)
		# x: [batch_size, in_capsules=1152, out_capsules=10, in_capsule_size=8]

		
		W = torch.cat([self.W.unsqueeze(0)] * si, dim=0)#.to(device)
		# W: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_size=16, in_capsule_size=8]

		# Transform inputs by weight matrix `W`.
		u_hat = torch.matmul(W, x.unsqueeze(4)) 
		# matrix multiplication
		# u_hat: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_size=16, 1]

		u_hat_detached = u_hat.detach()
		# u_hat_detached: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_size=16, 1]
		# In forward pass, `u_hat_detached` = `u_hat`, and 
		# in backward, no gradient can flow from `u_hat_detached` back to `u_hat`.

		# Initialize routing logits to zero. #if self.gpu >= 0:
		b_ij = Variable(torch.zeros(self.in_capsules, self.out_capsules, 1)).to(device)
		# b_ij: [in_capsules=1152, out_capsules=10, 1]
		#pdb.set_trace()

		#ルーティング
		for iteration in range(self.routing_iters):
			# Convert routing logits to softmax.
			#pdb.set_trace()

			c_ij = b_ij.unsqueeze(0)
			c_ij = c_ij.log_softmax(dim=2)

			c_ij = torch.cat([c_ij] * si, dim=0).unsqueeze(4)
			# c_ij: [batch_size, in_capsules=1152, out_capsules=10, 1, 1]
			# 3イタレーション行う
			if iteration == self.routing_iters - 1:
				# Apply routing `c_ij` to weighted inputs `u_hat`.

				s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # element-wise product
				# s_j: [batch_size, 1, out_capsules=10, out_capsule_size=16, 1]
				v_j = s_j.clone()
				# v_j: [batch_size, 1, out_capsules=10, out_capsule_size=16, 1]

			else:
				# Apply routing `c_ij` to weighted inputs `u_hat`.
				s_j = (c_ij * u_hat_detached).sum(dim=1, keepdim=True)
				# s_j: [batch_size, 1, out_capsules=10, out_capsule_size=16, 1]
				v_j = squash(s_j, dim=3)
				# v_j: [batch_size, 1, out_capsules=10, out_capsule_size=16, 1]
				# u_hat_detached: [batch_size, in_capsules=1152, out_capsules=10, out_capsule_size=16, 1]
				# Compute inner products of 2 16D-vectors, `u_hat` and `v_j`.

				u_vj1 = torch.matmul(u_hat_detached.transpose(3, 4), v_j).squeeze(4).mean(dim=0, keepdim=False)#
				# u_vj1: [in_capsules=1152, out_capsules=10, 1]

				# Update b_ij (routing).
				b_ij = b_ij + u_vj1
		

		#del b_ij, u_vj1 ,u_hat,u_hat_detached ,c_ij,s_j
		v_j = torch.sqrt((v_j **2).sum(dim=3))
		return v_j.squeeze(3).squeeze(1) # [batch_size, out_capsules, out_capsule_size]