import json
import regex as re
import base64
from collections.abc import Iterable, Iterator
import time
import multiprocessing
import os
import tokenizer.bpe as bpe

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
REPLACEMENT_CHAR = '\ufffd'
NUM_PROCESSES = os.cpu_count()

class Tokenizer():

	def __init__(
		self, 
		vocab: dict[int, bytes], 
		merges: list[tuple[bytes, bytes]], 
		special_tokens: list[str] | None = None):

		self.vocab_token_to_bytes = vocab
		self.vocab_bytes_to_token = {value: key for key, value in vocab.items()}
		self.special_tokens = set(special_tokens) if special_tokens else None
		self.merges_list = merges
		self.merges_dict = {merges[i]: i for i in range(len(merges))}

	@classmethod
	def from_files(
		cls,
		vocab_filepath: str,
		merges_filepath: str,
		special_tokens: list[str] | None = None):
		"""Load tokenizer from vocab and merges files.

		Files must use base64 encoding for byte sequences.
		Claude wrote this function.
		"""
		# Load vocab: JSON file mapping token_id (str) -> base64-encoded bytes (str)
		with open(vocab_filepath, 'r', encoding='utf-8') as f:
			vocab_raw = json.load(f)
			vocab = {int(key): base64.b64decode(value) for key, value in vocab_raw.items()}

		# Load merges: text file with "base64_part1 + base64_part2" per line
		merges = []
		with open(merges_filepath, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if line:
					parts = line.split(' + ')
					merge = tuple(base64.b64decode(part) for part in parts)
					merges.append(merge)

		return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


	def pretokenize(self, text: str) -> list[str]:
		"""Chop up long string into 'words'.

		Split on special tokens first to keep them intact.
		"""
		if not self.special_tokens:
			return re.findall(PAT, text)

		special_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=lambda x: len(x), reverse=True))
		parts = re.split(f'({special_pattern})', text)
		
		result = []
		for part in parts:
			if part in self.special_tokens:
				result.append(part)
			elif part:
				result.extend(re.findall(PAT, part))
		
		return result


	def bytify(self, words: list[str]) -> list[tuple[bytes, bytes] | str]:
		"""Go from word to bytes.

		Keep the special tokens intact.

		Example: the -> (b't', b'h', b'e')
		Example: <|endoftext|> -> b'<|endoftext|>'
		"""
		results = []
		for word in words:
			# word = the
			# word_encoded = b'the'
			# word_bytes = (b't', b'h', b'e')
			word_encoded = word.encode('utf-8')
			if self.special_tokens and word in self.special_tokens:
				word_bytes = bytes(word_encoded)
			else:
				word_bytes = tuple(bytes([b]) for b in word_encoded)
			results.append(word_bytes)
		return results


	def pairs(self, word: tuple[bytes, bytes]) -> list[tuple[bytes, bytes]]:
		"""Find pairs in word.


		Example: (b't', b'h', b'e') -> [(b't', b'h'), (b'h', b'e')]
		"""
		return [word[i:i+2] for i in range(len(word)-1)]


	def merge(self, word: tuple[bytes, bytes], merge_pair: tuple[bytes, bytes]) -> tuple[bytes, bytes]:
		"""Return the same word but with some letters merged."""
		merged = b''.join(merge_pair)

		new_word = []
		i = 0
		while i < len(word):
			if word[i:i+2] == merge_pair:
				new_word.append(merged)
				i += 2
			else:
				new_word.append(word[i])
				i += 1
		
		return tuple(new_word)


	def keep_merging(self, word: tuple[bytes, bytes] | bytes) -> tuple[bytes,bytes]:
		"""Keep merging letters until we can't anymore."""
		if isinstance(word, bytes):
			return word

		while True:
			pairs = self.pairs(word)
			if len(pairs) == 0:
				return word
			merge_order = [self.merges_dict[pair] if pair in self.merges_dict else float('inf') for pair in pairs]
			min_merge = min(merge_order)
			if min_merge == float('inf'):
				return word
			index_min_merge = merge_order.index(min_merge)
			merge_pair = self.merges_list[min_merge]
			word = self.merge(word, merge_pair)

	@property
	def endoftext_token(self) -> int:
		"""Return the token that represents end of text."""
		return self.vocab_bytes_to_token[bpe.END_OF_TEXT.encode('utf-8')]


	def tokenize(self, word: tuple[bytes, bytes] | bytes) -> list[int]:
		"""Bytes to tokens.

		For special tokens (b'<|endoftext|>') directly try to lookup.
		"""
		if word in self.vocab_bytes_to_token:
			return [self.vocab_bytes_to_token[word]]
		return [self.vocab_bytes_to_token[b] for b in word]


	def parallel_encode(self, texts: list[str]) -> list[list[str]]:
		with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
			encoded_texts = pool.map(self.encode, texts)
			return encoded_texts


	def encode(self, text: str) -> list[int]:
		"""Encode the text.

		Input: string
		Output: tokens
		"""
		# start = time.time()

		tokens = []

		pretokenized = self.pretokenize(text)
		bytified = self.bytify(pretokenized)

		for word in bytified:
			merged_word = self.keep_merging(word)
			tokenized = self.tokenize(merged_word)
			tokens.extend(tokenized)

		# total_time = time.time() - start
		# original_length = len(text.encode("utf-8"))
		# print(f'Speed: {original_length/total_time}')
		return tokens


	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
		"""Lazy encoder."""
		for text in iterable:
			encoded = self.encode(text)
			for token in encoded:
				yield token


	def decode(self, ids: list[int]) -> str:
		"""Go from tokens -> string."""
		replacement_char_bytes = REPLACEMENT_CHAR.encode('utf-8')
		text_bytes = [self.vocab_token_to_bytes[token] if token in self.vocab_token_to_bytes else replacement_char_bytes for token in ids]
		together = b''.join(text_bytes)
		return together.decode('utf-8', errors='replace')


