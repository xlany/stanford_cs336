import torch
from einops import reduce
import math


def softmax(x: torch.Tensor, dim_i: int):
	"""Apply softmax by fixing dimension i."""
	# x is [batch_size, sequence_len]
	# Normalize everything along dim_i
	# If dim_i = 0 (for every row...), normalize all the cols
	# If dim_i = 1 (for every col...), normalize all the rows
	# Lets say we were using dim_i = 1
	v_max = torch.max(x, dim=dim_i, keepdim=True)[0]	# Get the max of each row [batch_size 1]
														# 	keepdim=True basically sets the dimension (i.e. sequence_len) to [... 1]
	v = x - v_max										# Broadcast will subtract the max of each row from every element in that row
														#	So there will be an element of each row that is 0
														#	[batch_size, sequence_len] - [batch_size 1] = [batch_size, sequence_len]
	exp_v = torch.exp(v)								# Element-wise [batch_size, sequence_len]
	denom = exp_v.sum(dim=dim_i, keepdim=True)			# Sum across each row [batch_size, 1]
	return exp_v / denom								# Broadcast will divide [batch_size, sequence_len] / [batch_size, 1] = [batch_size, sequence_len]


	# This processes a single slice but I need to process all the slices
	# v = x[dim_i]
	# v = v - torch.max(v)
	# exp_v = torch.exp(v)
	# denom = reduce(exp_v, 'n -> ', 'sum')   # Go from vector of size [n] to scalar by summing
	# softmax_dim_i = exp_v / denom			# Element-wise divide
	# x[dim_i] = softmax_dim_i
	# return x

	"""
	Resource accounting
		Parameters = 0
		Activations = batch_size * sequence_len
		FLOPs = 5 * batch_size * sequence_len

	Activations accounting
		Reuse memory for the intermediate operations
		Total = batch_size * sequence_len

	FLOPs accounting
		v_max = batch_size * sequence_len
		subtraction = batch_size * sequence_len
		exp = batch_size * sequence_len
		sum = batch_size * sequence_len
		division = batch_size * sequence_len
	"""

def neg_log_softmax(x: torch.Tensor, dim_i: int):
	"""Calc -log(softmax) but with some nice subtraction for numerical stability.

	softmax[i] = exp(x[i])/sum(exp(x))
	log(softmax[i]) = log(exp(x[i])) - log(sum(exp(x)))
	-log(softmax[i]) 
		= log(sum(exp(x))) - log(exp(x[i]))  but the log & exp cancel!
		= log(sum(exp(x))) - x[i]
	"""
	v_max = torch.max(x, dim=dim_i, keepdim=True)[0]
	v = x - v_max
	exp_v = torch.exp(v)
	return torch.log(exp_v.sum(dim=dim_i, keepdim=True)) - v




