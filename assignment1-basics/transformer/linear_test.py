import torch
import transformer.linear as linear

def test_linear():
	d_in = 2
	d_out = 1
	linear_layer = linear.Linear(in_features=d_in, out_features=d_out)

	batch_size = 3
	sequence_length = 4
	x_shape = (batch_size, sequence_length, d_in)
	fill_value = 1.0
	tensor_3d = torch.full(x_shape, fill_value)

	forward_pass = linear_layer.forward(tensor_3d)
	assert forward_pass.shape == (batch_size, sequence_length, d_out)

