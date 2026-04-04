import torch 
import tokenizer.bpe_encoding as tokenizer_lib
from tokenizer import bpe
import bisect 

import cs336_basics.transformer.transformer as trans
import cs336_basics.training.checkpointing as checkpointing

import einops

# VOCAB_FILEPATH = './tokenizer_training_encoding/train_bpe_tinystories_vocab.txt'
# MERGES_FILEPATH = './tokenizer_training_encoding/train_bpe_tinystories_merges.txt'
VOCAB_FILEPATH = './tokenizer_training_encoding/train_bpe_owt_train_vocab.txt'
MERGES_FILEPATH = './tokenizer_training_encoding/train_bpe_owt_train_merges.txt'
CHECKPOINT_FILEPATH = './checkpoints'
LOG_PER_TOKEN = 10

def probability_distr(
	v: torch.Tensor,
	temperature: float | None = None,
	top_p: float | None = None,
) -> torch.Tensor:
	"""Convert logits into probability distr.
	
	Vanilla: exp(v) / sum(exp(v))
	Temperature: exp(v / temp) / sum(exp(v / temp))
	Top_p: sort by the largest probabilities and find the cutoff point where 
		left = tokens to include
		right = tokens to set probability = 0
		Perform the zero-ing out with a mask and re-normalized to total probabiliy = 1.0
	"""
	if temperature:
		v = torch.divide(v, temperature)
	exp_v = torch.exp(v)
	denom = exp_v.sum()
	distr = exp_v / denom
	if top_p:
		if top_p == 1:
			return distr
		sorted_probs, sorted_indices = torch.sort(distr, descending=True)
		cumsum_probs = torch.cumsum(sorted_probs, dim=0)
		cutoff_index = bisect.bisect_right(cumsum_probs.tolist(), top_p)
		cutoff_index = max(cutoff_index, 1)
		mask = torch.ones_like(distr)
		mask[sorted_indices[cutoff_index:]] = 0.0 
		distr = distr * mask
		return distr / torch.sum(distr)
	return distr

class LLM():
	"""Large language model used to generate text."""
	def __init__(
			self, 
			model: torch.nn.Module,
			tokenizer: tokenizer_lib.Tokenizer,
			device: torch.device | None = None
		) -> None:
		self.model = model
		self.tokenizer = tokenizer
		self.device = device

	def generate(
			self,
			prompt: str,
			temperature: float | None = None,
			top_p: float | None = None,
			max_tokens: int | None = 100,
		) -> str:
		"""Rollout prompt up to max_tokens long.
		
		Can terminate early if model outputs END_OF_TEXT token.
		"""
		# Prompt -> input tokens
		prompt_tokens: list[int] = self.tokenizer.encode(prompt)
		prompt_tensor: torch.Tensor = torch.tensor(prompt_tokens, device=self.device)
		prompt_tensor = einops.rearrange(prompt_tensor, 'n -> 1 n')  # Turn from [sequence_len] into [batch_size, sequence_len] shape
		
		output_tokens: list[int] = []
		# Autoregressive next token prediction
		self.model.eval()
		with torch.no_grad():
			for i in range(max_tokens):
				if i % LOG_PER_TOKEN == 0: 
					print(f'Generating {i}th token...')
				logits = self.model.forward(prompt_tensor)  # [batch_size, sequence_len, vocab_size]
				next_token_logits = logits[0, -1, :]  # [vocab_size] for the last word in sequence_len
				probs = probability_distr(  # Convert into probability distr that sum to 1 across all words in vocab_size
					v=next_token_logits, 
					temperature=temperature, 
					top_p=top_p
				) 
				sample = torch.multinomial(probs, num_samples=1)  # Sample token, weighed by probability
				prompt_tensor = torch.cat([prompt_tensor, einops.rearrange(sample, 'n -> 1 n')], dim=-1)  # Concatinate to the running prompt
				output_tokens.extend(sample.tolist())  # Track the generated tokens
				if sample.item() == bpe.END_OF_TEXT_TOKEN:
					break
		print(f'Generated {len(output_tokens)} tokens.')
		output_str = ''.join(self.tokenizer.decode(output_tokens))
		print(prompt + output_str)
		return output_str


if __name__=='__main__':
	run_name = 'openwebtext'

	device = torch.device("mps")

	# Initialize model with trained weights
	model = trans.Transformer(
		vocab_size=10_000,
		context_length=256,
		d_model=512,
		d_ff=1344,
		num_layers=4,
		num_heads=16,
		rope_theta=4_000,
		device=device,
		dtype=torch.float32,
	)
	model = torch.compile(model, backend="aot_eager")

	checkpointing.load_latest_checkpoint(
		folder=f'{CHECKPOINT_FILEPATH}/{run_name}/',
		model=model,
	)

	# Initialize tokenizer
	tokenizer = tokenizer_lib.Tokenizer.from_files(
		vocab_filepath=VOCAB_FILEPATH,
		merges_filepath=MERGES_FILEPATH,
		special_tokens=[bpe.END_OF_TEXT],
	)

	# Put it all into LLM class
	llm = LLM(
		model=model, 
		tokenizer=tokenizer, 
		device=device,
	)
	llm.generate(
		prompt="Once upon a time...",
		temperature=0.7,
		top_p=0.95,
		max_tokens=300,
	)
	