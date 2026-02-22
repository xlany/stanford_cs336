import torch
import transformer.rmsnorm as rmsnorm
import transformer.attention as attention
import transformer.rope as rope
import transformer.positionwise_feedforward as positionwise_feedforward
import transformer.linear as linear
import transformer.embedding as embedding
import transformer.softmax as softmax

class TransformerBlock():

	def __init__(
		self, 
		d_model: int, 
		num_heads: int, 
		d_ff: int,
		rope_config: rope.RopeConfig | None = None,
		) -> None:
		self.rmsnorm1 = rmsnorm.RMSNorm(d_model)
		self.rmsnorm2 = rmsnorm.RMSNorm(d_model)
		self.multihead_self_attention = attention.CausalMultiHeadSelfAttention(d_model, num_heads, rope_config)
		self.ffn = positionwise_feedforward.SwiGLU(d_model, d_ff)

	def from_weights(self, weights: dict[str, torch.Tensor]) -> None:
		"""Initialize from weights."""
		self.multihead_self_attention.from_weights(
			weights['attn.q_proj.weight'],
			weights['attn.k_proj.weight'],
			weights['attn.v_proj.weight'],
			weights['attn.output_proj.weight'],
			)

		self.rmsnorm1.load_state_dict({'g': weights['ln1.weight']})
		self.rmsnorm2.load_state_dict({'g': weights['ln2.weight']})
		self.ffn.load_state_dict(
			{
				'W1.W': weights['ffn.w1.weight'],
				'W2.W': weights['ffn.w2.weight'],
				'W3.W': weights['ffn.w3.weight'],
			})

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Implement Figure 2 in assignment.

		Resource accounting
			Parameters = 4 * d_model * d_model
					   + 3 * d_model * d_ff
					   + 2 * d_model 
			Activations = 13 * batch_size * sequence_length * d_model + 2 * batch_size * sequence_length * d_ff + 4 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v
			FLOPs = 22 * batch_size * sequence_len * d_model
				+ 8 * batch_size * sequence_len * d_model * d_model
				+ 4 * batch_size * sequence_len * sequence_len * d_model
				+ 7 * batch_size * sequence_len * sequence_len * num_heads 
				+ 6 * batch_size * sequence_length * d_model * d_ff
				+ 3 * batch_size * sequence_length * d_ff

		Parameters calculation
			rmsnorm1 = d_model
			rmsnorm2 = d_model
			multihead self attention = 4 * d_model * d_model
			ffn = 3 * d_model * d_ff
			Total = 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model

		Activations calculation
			rmsnorm1 = batch_size * sequence_len * d_model
			attention = 4 * batch_size * sequence_len * d_model + 2 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v
			rmsnorm2 = batch_size * sequence_len * d_model
			ffn = 2 * batch_size * sequence_len * d_ff + batch_size * sequence_len * d_model
			Total = 7 * batch_size * sequence_length * d_model + 2 * batch_size * sequence_length * d_ff + 2 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v

		FLOPs calculation
			rmsnorm1 = 4 * batch_size * sequence_len * d_model
			attention = 8 * batch_size * sequence_len * d_model * d_model
				  + 6 * batch_size * sequence_len * d_model
				  + 4 * batch_size * sequence_len * sequence_len * d_model
				  + 7 * batch_size * sequence_len * sequence_len * num_heads
			residual1 = batch_size * sequence_len * d_model
			rmsnorm2 = 4 * batch_size * sequence_len * d_model
			ffn = 6 * batch_size * sequence_length * d_model * d_ff + 3 * batch_size * sequence_length * d_ff
			residual2 = batch_size * sequence_len * d_model
			Total = 22 * batch_size * sequence_len * d_model
				+ 8 * batch_size * sequence_len * d_model * d_model
				+ 4 * batch_size * sequence_len * sequence_len * d_model
				+ 7 * batch_size * sequence_len * sequence_len * num_heads 
				+ 6 * batch_size * sequence_length * d_model * d_ff
				+ 3 * batch_size * sequence_length * d_ff
		"""
		attention_output = self.multihead_self_attention.forward(self.rmsnorm1.forward(x))
		layer1 = x + attention_output

		ffn_output = self.ffn.forward(self.rmsnorm2.forward(layer1))
		return layer1 + ffn_output


class Transformer():

	def __init__(
		self,
	    vocab_size: int,
	    context_length: int,
	    d_model: int,
	    num_layers: int,
	    num_heads: int,
	    d_ff: int,
	    rope_theta: float,
		) -> None:
		self.num_layers = num_layers
		self.token_embedding = embedding.Embedding(vocab_size, d_model)
		rope_config = rope.RopeConfig(theta=rope_theta, max_seq_len=context_length, d_k=int(d_model / num_heads))
		self.transformer_blocks = [TransformerBlock(d_model, num_heads, d_ff, rope_config) for _ in range(num_layers)]
		self.rmsnorm = rmsnorm.RMSNorm(d_model)
		self.output_embedding = linear.Linear(d_model, vocab_size)


	def from_weights(self, weights: dict[str, torch.Tensor]) -> None:
		"""Initialize from weights."""
		self.token_embedding.load_state_dict({'W': weights['token_embeddings.weight']})
		self.rmsnorm.load_state_dict({'g': weights['ln_final.weight']})
		self.output_embedding.load_state_dict({'W': weights['lm_head.weight']})
		for num_layer in range(self.num_layers):
			layer_prefix = f'layers.{num_layer}.'
			transformer_block_weights = {k[len(layer_prefix):] : v for k, v in weights.items() if k.startswith(layer_prefix)}
			self.transformer_blocks[num_layer].from_weights(transformer_block_weights)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Implement Figure 1 in assignment.

		Resource accounting
			Parameters = 2 * vocab_size * d_model 
					   + 4 * num_layers * d_model * d_model 
					   + 3 * num_layers * d_model * d_ff 
					   + (2 * num_layer + 1) * d_model
			Activations = num_layers * (7 * batch_size * sequence_length * d_model + 2 * batch_size * sequence_length * d_ff + 2 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v)
						+ batch_size * sequence_length * d_model
						+ batch_size * sequence_length * vocab_size
			FLOPs = (22 * num_layers + 4) * batch_size * sequence_len * d_model
				+ 8 * num_layers * batch_size * sequence_len * d_model * d_model
				+ 4 * num_layers * batch_size * sequence_len * sequence_len * d_model
				+ 7 * num_layers * batch_size * sequence_len * sequence_len * num_heads 
				+ 6 * num_layers * batch_size * sequence_length * d_model * d_ff
				+ 3 * num_layers * batch_size * sequence_length * d_ff
				+ 4 * batch_size * sequence_len * vocab_size * d_model

		Parameter calculation
			token embedding = vocab_size * d_model
			transformer blocks = num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
			rmsnorm = d_model
			output embedding = vocab_size * d_model
			Total = 2 * vocab_size * d_model + 4 * num_layers * d_model * d_model + 3 * num_layers * d_model * d_ff + (2 * num_layer + 1) * d_model

		Activation calculations
			Simplified according to instructions on pg 33
			transformer_blocks = num_layers * (7 * batch_size * sequence_length * d_model + 2 * batch_size * sequence_length * d_ff + 2 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v)
			rms_norm = batch_size * sequence_length * d_model
			output_embedding = batch_size * sequence_length * vocab_size

		FLOPs calculation
			token embedding = [batch_size sequence_len vocab_size] x [vocab_size d_model]
				= 2 * batch_size * sequence_len * vocab_size * d_model
			num_layers x transformer block 
				= num_layers * (22 * batch_size * sequence_len * d_model
				+ 8 * batch_size * sequence_len * d_model * d_model
				+ 4 * batch_size * sequence_len * sequence_len * d_model
				+ 7 * batch_size * sequence_len * sequence_len * num_heads 
				+ 6 * batch_size * sequence_length * d_model * d_ff
				+ 3 * batch_size * sequence_length * d_ff)
			rmsnorm = 4 * batch_size * sequence_len * d_model
			output embedding = [batch_size sequence_len d_model] x [d_model vocab_size]
				= 2 * batch_size * sequence_len * vocab_size * d_model 
				Which makes sense b/c it's the inverse of the token embedding layer
			Total = (22 num_layers + 4) * batch_size * sequence_len * d_model
				+ 8 num_layers * batch_size * sequence_len * d_model * d_model
				+ 4 num_layers * batch_size * sequence_len * sequence_len * d_model
				+ 7 num_layers * batch_size * sequence_len * sequence_len * num_heads 
				+ 6 num_layers * batch_size * sequence_length * d_model * d_ff
				+ 3 num_layers * batch_size * sequence_length * d_ff
				+ 4 * batch_size * sequence_len * vocab_size * d_model

		"""
		transformer_input = self.token_embedding.forward(x)
		for transformer_block in self.transformer_blocks:
			transformer_input = transformer_block.forward(transformer_input)
		normalized_transformer_output = self.rmsnorm.forward(transformer_input)
		output_embedding = self.output_embedding.forward(normalized_transformer_output)
		return output_embedding


