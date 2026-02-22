"""The final numbers in this file are SUPER off compared to industry.

I calculated them relative to my own implementation of the assignment as practice.
I cannot figure out why it's so off compared to industry without deviating from my implementation.
Going to move on. No need to get blocked here.
"""
import math

def bytes_to_gb(b: int) -> int:
	"""Convert bytes to GB"""
	return b / (10 ** 9)

def gb_to_bytes(gb: int) -> int:
	"""Convert GB to bytes"""
	return gb * (10 ** 9) 

class ModelResourceAccounting():
	def __init__(
		self,
		model_name: str,
		vocab_size: int,
		context_length: int,
		num_layers: int,
		d_model: int,
		num_heads: int,
		d_ff: int,
		batch_size: int = 1
		) -> None:
		self.model_name = model_name
		self.vocab_size = vocab_size
		self.context_length = context_length
		self.sequence_len = context_length
		self.num_layers = num_layers
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_ff = d_ff
		self.batch_size = batch_size

	def block_memory(self, n_blocks: int = 1) -> int:
		"""The most frequent unit I see is batch_size * sequence_len * d_model."""
		b = n_blocks * self.batch_size * self.sequence_len * self.d_model * 4
		gb = bytes_to_gb(b)
		print(f'{n_blocks} block memory for {self.model_name} is {gb} GB')
		return gb

	def calc_parameters(self) -> int:
		"""Calculate transformer parameters."""
		parameters = 2 * self.vocab_size * self.d_model \
		+ 4 * self.num_layers * self.d_model * self.d_model \
		+ 3 * self.num_layers * self.d_model * self.d_ff \
		+ (2 * self.num_layers + 1) * self.d_model
		print(f'Model {self.model_name} parameters: {parameters}')

		memory_bytes = parameters * 4 
		memory_gb = memory_bytes / (10 ** 9)
		print(f'Model {self.model_name} takes memory: {memory_gb} GBs')
		return parameters

	def calc_parameters_adamw(self) -> int:
		"""AdamW parameters are 2x the model parameters."""
		return 2 * self.calc_parameters()

	def calc_activations(self) -> int:
		sequence_len_q = self.sequence_len
		sequence_len_k = self.sequence_len
		d_v = int(self.d_model / self.num_heads)

		transformer_blocks = self.num_layers * (7 * self.batch_size * self.sequence_len * self.d_model + 2 * self.batch_size * self.sequence_len * self.d_ff + 2 * self.num_heads * self.batch_size * sequence_len_q * sequence_len_k + self.num_heads * self.batch_size * sequence_len_q * d_v)
		rms_norm = 3 * self.batch_size * self.sequence_len * self.d_model
		output_embedding = self.batch_size * self.sequence_len * self.vocab_size

		return transformer_blocks + rms_norm + output_embedding

	def calc_peak_memory(self) -> int:
		return self.calc_parameters() + self.calc_parameters_adamw() + self.calc_activations()

	def calc_batch_size(self, max_memory_gb: int) -> int:
		"""Given a max memory amount, what's the biggest batch_size we can train with? 

		Parameters don't depend on the batch size. Only activations do.

		max_memory_gb = activations + 3 * parameters 
		Put in the expanded form of activations and shuffle stuff around... I had Claude do this.

		max_memory_gb = a * batch_size + b
			a = num_layers * (13 * sequence_len * d_model + 2 * sequence_len * d_ff + 4 * num_heads * sequence_len_q * sequence_len_k + num_heads * sequence_len_q * d_v) + 3 * sequence_len * d_model + sequence_len * vocab_size
			b = 3 * parameters 

		"""
		max_memory_bytes = max_memory_gb * (10 ** 9) 
		max_memory_activation_bytes = max_memory_bytes - 4 * 3 * self.calc_parameters()
		max_memory_activation_params = max_memory_activation_bytes / 4

		sequence_len_q = self.sequence_len
		sequence_len_k = self.sequence_len
		d_v = int(self.d_model / self.num_heads)
		a = self.num_layers * (7 * self.sequence_len * self.d_model + 2 * self.sequence_len * self.d_ff + 2 * self.num_heads * sequence_len_q * sequence_len_k + self.num_heads * sequence_len_q * d_v) + self.sequence_len * self.d_model + self.sequence_len * self.vocab_size
		max_batch_size = math.floor(max_memory_activation_params / a)
		print(f'Max batch size for {self.model_name} and {max_memory_gb} GB memory is {max_batch_size}')
		return max_batch_size


	def calc_transformer_flops(self, announce: bool = True) -> int:
		flops = (22 * self.num_layers + 4) * self.batch_size * self.sequence_len * self.d_model \
		+ 8 * self.num_layers * self.batch_size * self.sequence_len * self.d_model * self.d_model \
		+ 4 * self.num_layers * self.batch_size * self.sequence_len * self.sequence_len * self.d_model \
		+ 7 * self.num_layers * self.batch_size * self.sequence_len * self.sequence_len * self.num_heads \
		+ 6 * self.num_layers * self.batch_size * self.sequence_len * self.d_model * self.d_ff \
		+ 3 * self.num_layers * self.batch_size * self.sequence_len * self.d_ff \
		+ 4 * self.batch_size * self.sequence_len * self.vocab_size * self.d_model
		if announce:
			print(f'Model {self.model_name} FLOPs: {flops}')
		return flops

	def calc_optimizer_flops(self) -> int:
		return 14 * self.batch_size * self.sequence_len * self.d_model

	def calc_days(self) -> int: 
		possible_flops_per_second = 19.5 / 2 * (10 ** 12)
		possible_flops_per_day = possible_flops_per_second * 60 * 60 * 24
		flops_per_step = 3 * (self.calc_transformer_flops() + self.calc_optimizer_flops())
		total_flops = 400_000 * flops_per_step
		days = int(total_flops / possible_flops_per_day)
		print(f'It would take {days} days on a single A100')
		return days


	def _flops_attention(self) -> int:
		flops = 8 * self.batch_size * self.sequence_len * self.d_model * self.d_model \
		+ 6 * self.batch_size * self.sequence_len * self.d_model \
		+ 4 * self.batch_size * self.sequence_len * self.sequence_len * self.d_model \
		+ 7 * self.batch_size * self.sequence_len * self.sequence_len * self.num_heads
		flops = flops * self.num_layers
		# print(f'Model {self.model_name} attention FLOPs: {flops}')
		return flops

	def _flops_ffn(self) -> int:
		flops = 6 * self.batch_size * self.sequence_len * self.d_model * self.d_ff \
		+ 3 * self.batch_size * self.sequence_len * self.d_ff
		flops = flops * self.num_layers
		# print(f'Model {self.model_name} ffn FLOPs: {flops}')
		return flops

	def _flops_attention_percentage(self) -> float:
		percentage = self._flops_attention() / self.calc_transformer_flops(announce=False)
		print(f'Model {self.model_name} attention FLOPs percentage: {percentage}')
		return percentage

	def _flops_ffn_percentage(self) -> float:
		percentage = self._flops_ffn() / self.calc_transformer_flops(announce=False)
		print(f'Model {self.model_name} ffn FLOPs percentage: {percentage}')
		return percentage

	def flops_percentage(self) -> None:
		flops_ffn_percentage = self._flops_ffn_percentage()
		flops_attention_percentage = self._flops_attention_percentage()
		everything_else = 1 - flops_ffn_percentage - flops_attention_percentage
		print(f'Model {self.model_name} everything else FLOPs percentage: {everything_else}')



if __name__=="__main__":

	gpt2_xl = ModelResourceAccounting(
		model_name='gpt2_xl',
		vocab_size=50_257,
		context_length=1_024,
		num_layers=48,
		d_model=1_600,
		num_heads=25,
		d_ff=6_400)

	gpt2_xl.calc_parameters()
	gpt2_xl.calc_transformer_flops()
	gpt2_xl.flops_percentage()

	"""
	(a) Consider GPT-2 XL, which has the following configuration:
	vocab_size : 50,257 
	context_length : 1,024
	num_layers : 48 
	d_model : 1,600
	num_heads : 25 
	d_ff : 6,400

	Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

	Transformer parameters
		= 2 * vocab_size * d_model 
		+ 4 * num_layers * d_model * d_model 
		+ 3 * num_layers * d_model * d_ff 
		+ (2 * num_layer + 1) * d_model

		= 2 * 50,257 * 1,600
		+ 4 * 48 * 1,600 * 1,600
		+ 3 * 48 * 1,600 * 6,400
		+ (2 * 48 + 1) * 1,600

		= 2,127,057,600
		= 2B parameter model

	Single precision floating point = float FP32 = 4 bytes
	Memory (rough) = 2B parameters * 4 bytes each 
		= 8B bytes
		= 8M KB
		= 8000 MB
		= 8 GB
	Memory (exact) = 2,127,057,600 parameters * 4 bytes each
		= 8,508,230,400 bytes
		= 8.5 GB




	(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

	Transformer forward pass FLOPs
		= (22 * num_layers + 4) * batch_size * sequence_len * d_model
		+ 8 * num_layers * batch_size * sequence_len * d_model * d_model
		+ 4 * num_layers * batch_size * sequence_len * sequence_len * d_model
		+ 7 * num_layers * batch_size * sequence_len * sequence_len * num_heads 
		+ 6 * num_layers * batch_size * sequence_length * d_model * d_ff
		+ 3 * num_layers * batch_size * sequence_length * d_ff
		+ 4 * batch_size * sequence_len * vocab_size * d_model

		= (22 * 48 + 4) * 1 * 1,024 * 1,600
		+ 8 * 48 * 1 * 1,024 * 1,600 * 1,600
		+ 4 * 48 * 1 * 1,024 * 1,024 * 1,600
		+ 7 * 48 * 1 * 1,024 * 1,024 * 25 
		+ 6 * 48 * 1 * 1,024 * 1,600 * 6,400
		+ 3 * 48 * 1 * 1,024 * 6,400
		+ 4 * 1 * 1,024 * 50,257 * 1,600

		= 4.6895071e+12
		= 4.69 trillion

	(c) Based on your analysis above, which parts of the model require the most FLOPs?

	It's within the transformer block but let us look at what part.
		attention 
			= 8 * batch_size * sequence_len * d_model * d_model
			+ 6 * batch_size * sequence_len * d_model
			+ 4 * batch_size * sequence_len * sequence_len * d_model
			+ 7 * batch_size * sequence_len * sequence_len * num_heads

			= 8 * 1 * 1,024 * 1,600 * 1,600
			+ 6 * 1 * 1,024 * 1,600
			+ 4 * 1 * 1,024 * 1,024 * 1,600
			+ 7 * 1 * 1,024 * 1,024 * 25

			= 27,875,737,600

		num_layers * attention as percentage of total FLOPs
			= (27,875,737,600 * 48) / 4.6895071e+12
			= 28.5%

		ffn 
			= 6 * batch_size * sequence_length * d_model * d_ff 
			+ 3 * batch_size * sequence_length * d_ff

			= 6 * 1 * 1,024 * 1,600 * 6,400 
			+ 3 * 1 * 1,024 * 6,400

			= 62,934,220,800

		num_layers * ffn as percentage of total FLOPs
			= (62,934,220,800 * 48) / 4.6895071e+12
			= 64.4%

		Everything else
			= 100 - 28.5 - 64.4
			= 7.1%

	Surprisingly the FFN takes a lot more compute than attention! Wow!
	"""

	gpt2_small = ModelResourceAccounting(
		model_name='gpt2_small',
		vocab_size=50_257,
		context_length=1_024,
		num_layers=12,
		d_model=768,
		num_heads=12,
		d_ff=6_400)

	gpt2_small.flops_percentage()


	gpt2_medium = ModelResourceAccounting(
		model_name='gpt2_medium',
		vocab_size=50_257,
		context_length=1_024,
		num_layers=24,
		d_model=1_024,
		num_heads=16,
		d_ff=6_400)

	gpt2_medium.flops_percentage()


	gpt2_large = ModelResourceAccounting(
		model_name='gpt2_large',
		vocab_size=50_257,
		context_length=1_024,
		num_layers=36,
		d_model=1_280,
		num_heads=20,
		d_ff=6_400)

	gpt2_large.flops_percentage()

	gpt2_xl.flops_percentage()

	"""
	(d) Repeat your analysis with 
		GPT-2 small (12 layers, 768 d_model, 12 heads), 
		GPT-2 medium (24 layers, 1024 d_model, 16 heads), and 
		GPT-2 large (36 layers, 1280 d_model, 20 heads). 
		As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

	Model gpt2_small ffn FLOPs percentage: 0.5861788411176935
	Model gpt2_small attention FLOPs percentage: 0.15801276346901935
	Model gpt2_small everything else FLOPs percentage: 0.25580839541328715

	Model gpt2_medium ffn FLOPs percentage: 0.6487781538266688
	Model gpt2_medium attention FLOPs percentage: 0.20950035533102074
	Model gpt2_medium everything else FLOPs percentage: 0.14172149084231045

	Model gpt2_large ffn FLOPs percentage: 0.6570123312740036
	Model gpt2_large attention FLOPs percentage: 0.2472069825910308
	Model gpt2_large everything else FLOPs percentage: 0.0957806861349656

	Model gpt2_xl ffn FLOPs percentage: 0.6441705959791045
	Model gpt2_xl attention FLOPs percentage: 0.2853253806099262
	Model gpt2_xl everything else FLOPs percentage: 0.07050402341096929

	As model gets bigger:
		Attention FLOPs share gets bigger
		Everything else FLOPs share gets smaller
		FFN FLOPs about the same
	"""

	gpt2_xl_long_context = ModelResourceAccounting(
		model_name='gpt2_xl_long_context',
		vocab_size=50_257,
		context_length=16_384,
		num_layers=48,
		d_model=1_600,
		num_heads=25,
		d_ff=6_400)

	gpt2_xl_long_context.calc_parameters()
	gpt2_xl_long_context.calc_transformer_flops()
	gpt2_xl_long_context.flops_percentage()

	gpt2_xl.calc_parameters()
	gpt2_xl.calc_transformer_flops()
	gpt2_xl.flops_percentage()

	"""
	(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

	Model gpt2_xl_long_context parameters: 2127057600
	Model gpt2_xl_long_context takes memory: 8.5082304 GBs
	Model gpt2_xl_long_context FLOPs: 154455454515200
	Model gpt2_xl_long_context ffn FLOPs percentage: 0.31292829201861233
	Model gpt2_xl_long_context attention FLOPs percentage: 0.6528219242064844
	Model gpt2_xl_long_context everything else FLOPs percentage: 0.034249783774903286

	Model gpt2_xl parameters: 2127057600
	Model gpt2_xl takes memory: 8.5082304 GBs
	Model gpt2_xl FLOPs: 4689507123200
	Model gpt2_xl ffn FLOPs percentage: 0.6441705959791045
	Model gpt2_xl attention FLOPs percentage: 0.2853253806099262
	Model gpt2_xl everything else FLOPs percentage: 0.07050402341096929

	For long context model:
		1. The attention FLOPs dominates rather than ffn FLOPs
		2. The total FLOPs increases by 154455454515200 / 4689507123200 = 32.93x even though we increased our context length by 16x 
	"""


	"""
	adamwAccounting
	(a) memory = activations + 3 * parameters 
		parameters = 2 * self.vocab_size * self.d_model \
			+ 4 * self.num_layers * self.d_model * self.d_model \
			+ 3 * self.num_layers * self.d_model * self.d_ff \
			+ (2 * self.num_layers + 1) * self.d_model
		activations = transformer_blocks + rms_norm + output_embedding
			transformer_blocks = self.num_layers * (7 * self.batch_size * self.sequence_len * self.d_model + 2 * self.batch_size * self.sequence_len * self.d_ff + 2 * self.num_heads * self.batch_size * sequence_len_q * sequence_len_k + self.num_heads * self.batch_size * self.sequence_len_q * d_v)
			rms_norm = self.batch_size * self.sequence_len * self.d_model
			output_embedding = self.batch_size * self.sequence_len * self.vocab_size

	(b) I got... max batch size 3?:( tried really hard but it's not getting bigger...
	"""
	gpt2_xl.calc_batch_size(max_memory_gb=80)
	"""
	(c) FLOPs per Adam activation = 14 * batch_size * sequence_len * d_model
								  = 0.0917504 GB
	"""
	gpt2_xl.block_memory(n_blocks=14)
	"""
	(d) It would take 6840 days on a single A100. But it cannot even fit in memory? 
		Something is wrong. I tried to figure out but couldn't. Moving on.
	"""
	gpt2_xl_1024 = ModelResourceAccounting(
		model_name='gpt2_xl',
		vocab_size=50_257,
		context_length=1_024,
		num_layers=48,
		d_model=1_600,
		num_heads=25,
		d_ff=6_400,
		batch_size=1024,
		)
	gpt2_xl_1024.calc_days()


