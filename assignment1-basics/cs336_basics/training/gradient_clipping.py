import torch
from collections.abc import Iterable


def clip_gradients(
	parameters: Iterable[torch.nn.Parameter], 
	max_l2_norm: float, 
	eps: float = 10 ** -6,
	) -> Iterable[torch.nn.Parameter]:
	"""If the gradients are too large overall, clip them to be close to max_l2_norm.
	
	This method modifies the gradients in-place and also returns them.
	"""
	grads = [parameter.grad for parameter in parameters]
	grads_tensor = torch.cat([grad.flatten() for grad in grads if grad is not None])
	l2_norm = torch.linalg.norm(grads_tensor)
	if l2_norm < max_l2_norm:
		return parameters

	ratio = max_l2_norm / (l2_norm + eps)
	for parameter in parameters:
		if parameter.grad is not None:
			parameter.grad = parameter.grad * ratio 
	return parameters