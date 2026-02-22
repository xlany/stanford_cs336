import math

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
	) -> float:
	"""Gradually build to the max_learning_rate, decrease it via cos schedule, and end up at min_learning_rate after that."""
	if it < warmup_iters:
		return it / warmup_iters * max_learning_rate

	if it > cosine_cycle_iters:
		return min_learning_rate

	return min_learning_rate + 1/2 * (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)