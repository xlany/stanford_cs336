import os
import collections
from typing import Any
import regex as re
import tokenizer.pretokenization as pretokenization
import multiprocessing

END_OF_TEXT = '<|endoftext|>'
INITIAL_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = os.cpu_count()

def pretokenize_and_count(chunk: list[str]) -> dict[tuple[bytes], int]:
    counts = collections.defaultdict(int)
    for d in chunk:
        pretokenized = re.findall(PAT, d)
        for word in pretokenized:
            encoded_word = word.encode('utf-8')
            counts[tuple(bytes([b]) for b in encoded_word)] += 1
    print(f'Counted {len(chunk)} words')
    return counts

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
            boundaries = pretokenization.find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")
            chunk_indicies: list[tuple[int, int]] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_indicies.append((start, end))
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

                mini_chunk = re.split(pattern, chunk)
                chunks.append([c for c in mini_chunk if c])

        return [c for c in chunks if c]

    def load_data(self) -> list[str]: 
        """Load data all at once."""
        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        with open(self.input_path, 'r') as f:
            data = f.read()
            chunks = re.split(pattern, data)
            return [chunk for chunk in chunks if chunk]
    
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
        with multiprocessing.Pool(processes=4) as pool:
            chunked_counts = pool.map(pretokenize_and_count, chunked_data)
        
        final_counts = collections.defaultdict(int)
        for chunked_count in chunked_counts:
            for key, value in chunked_count.items():
                final_counts[key] += value

        return final_counts
            
            
    def pretokenize(self, chunk: list[str]) -> list[str]:
        """Split long string into 'words'"""
        # return data.strip().split(" ")
        results: list[str] = []
        for d in chunk:
            results.extend(re.findall(PAT, d))
        print(f'Finished {len(results)} words in chunk')
        return results


    def count(self, chunk: list[str]) -> dict[tuple[bytes], int]:
        """Generate dictionary of {bytes: count}

        Encode the whole word before splitting to handle
        multi-byte elements (i.e. em-dash).
        """
        counts = collections.defaultdict(int)
        for word in chunk:
            encoded_word = word.encode('utf-8')
            counts[tuple(bytes([b]) for b in encoded_word)] += 1
        return counts
    
    def count_neighbors(self, data: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
        """Use sliding window of size 2 to count most frequent adjacent letters."""
        counts = collections.defaultdict(int)
        for word, count in data.items():
            word_len = len(word)
            for i in range(word_len-1):
                curr_window = word[i:i+2]
                counts[curr_window] += count
        return counts

    def max_count_neighbors(self, data: dict[tuple[bytes], int]) -> tuple[bytes]:
        """From map of counts, fetch the most frequent neighbor letters.

        Example: st
        
        If there are ties in the count, return the "biggest" letters. 
        """
        max_value = max(data.values())
        max_keys = [key for key, value in data.items() if value == max_value]
        return max(max_keys)

    def merge_data(self, data: dict[tuple[bytes], int], target_tuple: tuple[bytes], target_key: bytes, target_value: int) -> dict[tuple[bytes], int]:
        """Update the corpus with the new token.

        Example: most -> mo(st)
        """

        new_data = {}
        for word, value in data.items():
            new_word = []
            i = 0
            while i < len(word):
                curr_window = word[i:i+len(target_tuple)]
                if curr_window == target_tuple:
                    new_word.append(target_key)
                    i += len(target_tuple)
                else:
                    new_word.append(word[i])
                    i += 1
            new_data[tuple(new_word)] = value

        del data
        return new_data


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

        for i in range(self.num_new_words):
            if i % 100 == 0:
                print(f'On step {i}/{self.num_new_words}')

            # print(data)
            count_neighbors = self.count_neighbors(data)
            # print(count_neighbors)
            max_count_neighbors = self.max_count_neighbors(count_neighbors)
            # print(max_count_neighbors)
            merges.append(max_count_neighbors)
            # print(count_neighbors[max_count_neighbors])
            vocab_new_word = b''.join(max_count_neighbors)
            vocab_new_index = len(self.bytes_to_token)
            self.bytes_to_token[vocab_new_word] = vocab_new_index
            # print(self.bytes_to_token)
            
            new_data = self.merge_data(data, target_tuple=max_count_neighbors, target_key=vocab_new_word, target_value=vocab_new_index)
            # print(new_data)

            data = new_data

        return self.token_to_bytes, merges


    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.initialize_vocab()
        # All data at once
        # data = self.load_data()
        # pretokenized = self.pretokenize(data)
        # counts = self.count(pretokenized)

        # Chunked
        chunked_data = self.load_data_chunk()
        print(f'Chunked data: {len(chunked_data)} chunks')
        print(f'Documents: {sum([len(chunk) for chunk in chunked_data])} documents')
        # print(chunked_data)
        counts = self.pretokenize_and_count_parallel(chunked_data)
        # print(f'Gathered initial counts')
        return self.train_bpe(counts)

