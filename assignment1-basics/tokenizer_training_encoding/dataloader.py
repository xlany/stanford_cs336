import re
import os
import random
import tokenizer.pretokenization as pretokenization




class DataLoader:
	def __init__(self, input_path: str, special_tokens: list[str]):
		self.input_path = input_path
		self.special_tokens = special_tokens


	def sample_random_documents(self, k: int) -> list[str]:
		"""Sample k documents randomly with replacement."""
		data_chunks = self.load_data_chunk()

		documents: list[str] = []

		for i in range(k):
			chunk = random.choice(data_chunks)
			document = random.choice(chunk)
			documents.append(document)

		return documents

	def load(self) -> list[str]:
		with open(self.input_path, 'rb') as f:
			return f.read()


	def load_data_chunk(self, num_chunks: int) -> list[list[str]]:
		"""Load data in chunks according to END_OF_TEXT.
    
    We split on END_OF_TEXT but also need to keep it as part of the chunk.
    """
		pattern = "|".join(re.escape(token) for token in self.special_tokens)

		chunks: list[list[str]] = []
		with open(self.input_path, "rb") as f:
			boundaries = pretokenization.find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
			chunk_indicies: list[tuple[int, int]] = []
			for start, end in zip(boundaries[:-1], boundaries[1:]):
				chunk_indicies.append((start, end))
				f.seek(start)
				chunk = f.read(end - start).decode("utf-8", errors="ignore")

				mini_chunk = re.split(f'({pattern})', chunk)
				combined = [mini_chunk[i]+mini_chunk[i+1] for i in range(0, len(mini_chunk)-1, 2)]
				chunks.append([c for c in combined if c])

		return [c for c in chunks if c]
	
