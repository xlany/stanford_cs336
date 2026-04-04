import torch
import cs336_basics.training.optimizer as optimizer_lib
import cs336_basics.training.training_loop as training_loop

if __name__=='__main__':
	training_steps = 10

	print(f'Training with learning rate 1')
	weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
	optimizer = optimizer_lib.SGD([weights], lr=1)
	training_loop.training_loop(weights=weights, optimizer=optimizer, training_steps=training_steps)

	print(f'Training with learning rate 10')
	weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
	optimizer = optimizer_lib.SGD([weights], lr=10)
	training_loop.training_loop(weights=weights, optimizer=optimizer, training_steps=training_steps)

	print(f'Training with learning rate 100')
	weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
	optimizer = optimizer_lib.SGD([weights], lr=100)
	training_loop.training_loop(weights=weights, optimizer=optimizer, training_steps=training_steps)

	"""
	What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?

	Training with learning rate 1
	20.45394515991211
	19.64396858215332
	19.092281341552734
	18.65390968322754
	18.282697677612305
	17.957109451293945
	17.665067672729492
	17.39900779724121
	17.15381622314453
	16.925865173339844
	Steady decrease in loss, looks linear, has not saturated yet

	Training with learning rate 10
	27.328706741333008
	17.490371704101562
	12.89315414428711
	10.087516784667969
	8.170888900756836
	6.774602890014648
	5.713479042053223
	4.882330417633057
	4.2162766456604
	3.6728451251983643
	Large decrease in loss that becomes smaller decrease over time 

	Training with learning rate 100
	26.229711532592773
	26.229707717895508
	4.500305652618408
	0.10770244896411896
	1.3614110421801133e-16
	1.5173769103424003e-18
	5.1095421420017884e-20
	3.0437895862671605e-21
	2.6111591312338406e-22
	2.9012877833512175e-23
	First loss delta is consistently very small
		This is because we hella overshoot to the other side
	But then take HUGE strides
	By the end the loss deltas are very small and we have saturated
	"""