"""BPE

Optimizations:
1. Parallel load and initial count
2. Keep track of pairs (count_neighbors) and only update affected ones
3. Keep track of pair -> word mapping and only update affected ones
4. Tried minheap but it was 50% slower than vanilla
"""
import os
import collections
from typing import Any
import regex as re
import tokenizer.pretokenization as pretokenization
import multiprocessing
import time
# import heapq

END_OF_TEXT = '<|endoftext|>'
INITIAL_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)
NUM_PROCESSES = os.cpu_count()


def pretokenize_and_count(chunk: list[str]) -> dict[tuple[bytes], int]:
	counts = collections.defaultdict(int)
	for d in chunk:
		pretokenized = COMPILED_PAT.findall(d)
		for word in pretokenized:
			encoded_word = word.encode('utf-8')
			counts[tuple(bytes([b]) for b in encoded_word)] += 1
	print(f'Counted {len(chunk)} words')
	return counts


# class MaxHeap():
# 	def __init__(self):
# 		self.heap = []

# 	def initialize_heapq(self, count_neighbors: dict[tuple[bytes], int]):
# 		for word, value in count_neighbors.items():
# 			self.push(value, word)

# 	def push(self, value: int, word: tuple[bytes]):
# 		heapq.heappush(self.heap, (-value, ReversedBytes(word), word))

# 	def pop(self, count_neighbors: dict[tuple[bytes], int]):
# 		"""Lazily pop 

# 		Refresh numbers that are out of sync."""
# 		while True:
# 			max_value, _, max_word = heapq.heappop(self.heap)
# 			if max_word in count_neighbors:
# 				if count_neighbors[max_word] == -max_value:
# 					return max_word
# 				self.push(count_neighbors[max_word], max_word)


# class ReversedBytes:
#     """Custom comparator to use Python maxheap with bytes.

# 	There is no negative bytes so just need to reverse all the directions
# 	with a wrapper.
#     """
#     def __init__(self, value):
#         self.value = value
    
#     def __lt__(self, other):
#         return self.value > other.value
    
#     def __le__(self, other):
#         return self.value >= other.value
    
#     def __gt__(self, other):
#         return self.value < other.value
    
#     def __ge__(self, other):
#         return self.value <= other.value
    
#     def __eq__(self, other):
#         return self.value == other.value


class BytePairEncoderTokenizer():
	def __init__(self, 
				 input_path: str | os.PathLike,
				vocab_size: int,
				special_tokens: list[str]):
		self.input_path = input_path
		self.vocab_size = vocab_size
		self.special_tokens = special_tokens


	def load_data_chunk(self) -> list[list[str]]:
		"""Load data in chunks according to END_OF_TEXT."""
		pattern = "|".join(re.escape(token) for token in self.special_tokens)

		chunks: list[list[str]] = []
		with open(self.input_path, "rb") as f:
			boundaries = pretokenization.find_chunk_boundaries(f, NUM_PROCESSES*3, b"<|endoftext|>")
			chunk_indicies: list[tuple[int, int]] = []
			for start, end in zip(boundaries[:-1], boundaries[1:]):
				chunk_indicies.append((start, end))
				f.seek(start)
				chunk = f.read(end - start).decode("utf-8", errors="ignore")

				mini_chunk = re.split(pattern, chunk)
				chunks.append([c for c in mini_chunk if c])

		return [c for c in chunks if c]
	

	def initialize_vocab(self) -> None:
		"""Initializes the vocabulary of {bytes: token}"""
		self.bytes_to_token = {bytes([i]): i for i in range(INITIAL_VOCAB_SIZE)}
		for special_token in self.special_tokens:
			vocab_new_index = len(self.bytes_to_token)
			self.bytes_to_token[special_token.encode('utf-8')] = vocab_new_index


	@property
	def token_to_bytes(self):
		"""Reverse the vocab map."""
		return {value: key for key, value in self.bytes_to_token.items()}


	def pretokenize_and_count_parallel(self, chunked_data: list[list[str]]) -> dict[tuple[bytes], int]:
		"""Run the pretokenizer with multiprocessing."""
		with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
			chunked_counts = pool.map(pretokenize_and_count, chunked_data)
		
		final_counts = collections.defaultdict(int)
		for chunked_count in chunked_counts:
			for key, value in chunked_count.items():
				final_counts[key] += value

		return final_counts
			
	
	def count_neighbors(self, data: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
		"""Use sliding window of size 2 to count most frequent adjacent letters."""
		counts = collections.defaultdict(int)
		neighbors_to_words = collections.defaultdict(set)
		for word, count in data.items():
			word_len = len(word)
			for i in range(word_len-1):
				neighbor = (word[i], word[i+1])
				counts[neighbor] += count
				neighbors_to_words[neighbor].add(word)
		return counts, neighbors_to_words


	def max_count_neighbors(self, data: dict[tuple[bytes], int]) -> tuple[bytes]:
		"""From map of counts, fetch the most frequent neighbor letters.

		Example: st
		
		If there are ties in the count, return the "biggest" letters. 
		"""
		max_value = max(data.values())
		max_keys = [key for key, value in data.items() if value == max_value]
		return max(max_keys)


	def pairs(self, word: tuple[bytes]) -> list[tuple[bytes, bytes]]:
		"""For a given word (at bytes), return all the pairs inside."""
		return [(word[i], word[i+1]) for i in range(len(word)-1)]

	def merge_data(self, data: dict[tuple[bytes], int], count_neighbors: dict[tuple[bytes], int], neighbors_to_words: dict[tuple[bytes], set], target_tuple: tuple[bytes], target_key: bytes) -> dict[tuple[bytes], int]:
	# def merge_data(self, data: dict[tuple[bytes], int], count_neighbors: dict[tuple[bytes], int], neighbors_to_words: dict[tuple[bytes], set], heap: MaxHeap, target_tuple: tuple[bytes], target_key: bytes) -> dict[tuple[bytes], int]:
		"""Update the corpus with the new token.

		Directly update data and count_neighbors. Try to form the new word. 
		If the word changes, then decrement the old pairs & add the new pairs.

		Example: most -> mo(st)
		"""
		target_words = list(neighbors_to_words[target_tuple])
		# heap_updates = set()

		for word in target_words:
			value = data[word]

			new_word = []
			i = 0
			while i < len(word)-1:
				if (word[i], word[i+1]) == target_tuple:
					new_word.append(target_key)
					i += 2
					modified_word = True
				else:
					new_word.append(word[i])
					i += 1
					
			if i == len(word) - 1:
				new_word.append(word[i])

			new_word = tuple(new_word)
			old_pairs = self.pairs(word)
			new_pairs = self.pairs(new_word)

			for pair in old_pairs:
				count_neighbors[pair] -= value
				if count_neighbors[pair] == 0:
					del count_neighbors[pair]
				neighbors_to_words[pair].discard(word)
					
			for pair in new_pairs:
				count_neighbors[pair] += value
				neighbors_to_words[pair].add(new_word)
				# heap_updates.add(pair)

			data[new_word] = value
			del data[word] 

		# for pair in heap_updates:
		# 	heap.push(count_neighbors[pair], pair)

		return data, count_neighbors, neighbors_to_words


	@property
	def num_original_words(self) -> int:
		return INITIAL_VOCAB_SIZE + len(self.special_tokens)


	@property
	def num_new_words(self) -> int:
		return self.vocab_size - self.num_original_words
	   

	def new_words(self):
		"""Return the newly added words of the vocab."""
		new_words = [key for key, value in self.bytes_to_token.items() if value >= self.num_original_words]
		return new_words 
				

	def train_bpe(self, data: dict[tuple[bytes]]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
		"""
		Train the BPE

		Ideas for optimization: 
			Tree for vocab
			Some kinda max heap situation
		"""
		merges: list[tuple[bytes,bytes]] = []
		step_times = []

		count_neighbors, neighbors_to_words = self.count_neighbors(data)
		# heap = MaxHeap()
		# heap.initialize_heapq(count_neighbors)

		for i in range(self.num_new_words):
			start = time.time()
			# if i % 10 == 0:
			print(f'On step {i}/{self.num_new_words}')

			max_count_neighbors = self.max_count_neighbors(count_neighbors)
			# max_count_neighbors = heap.pop(count_neighbors)
			merges.append(max_count_neighbors)
			vocab_new_word = b''.join(max_count_neighbors)
			vocab_new_index = len(self.bytes_to_token)
			self.bytes_to_token[vocab_new_word] = vocab_new_index
			print('Merge')
			# data, count_neighbors, neighbors_to_words = self.merge_data(data, count_neighbors, neighbors_to_words, heap, target_tuple=max_count_neighbors, target_key=vocab_new_word)
			data, count_neighbors, neighbors_to_words = self.merge_data(data, count_neighbors, neighbors_to_words, target_tuple=max_count_neighbors, target_key=vocab_new_word)

			step_times.append(time.time() - start)

		print('Total step time: ', sum(step_times))
		return self.token_to_bytes, merges


	def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
		self.initialize_vocab()

		chunked_data = self.load_data_chunk()
		print(f'Chunked data: {len(chunked_data)} chunks')
		print(f'Documents: {sum([len(chunk) for chunk in chunked_data])} documents')
		counts = self.pretokenize_and_count_parallel(chunked_data)
		return self.train_bpe(counts)

