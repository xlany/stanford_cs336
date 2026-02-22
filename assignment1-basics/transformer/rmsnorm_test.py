import torch
import transformer.rmsnorm as rmsnorm

def test_rmsnorm():
	d_model = 2
	rmsnorm_layer = rmsnorm.RMSNorm(d_model=d_model)

	batch_size = 3
	sequence_length = 4
	x_shape = (batch_size, sequence_length, d_model)
	fill_value = 1.0
	x = torch.full(x_shape, fill_value)

	forward_pass = rmsnorm_layer.forward(x)
	assert forward_pass.shape == (batch_size, sequence_length, d_model)
