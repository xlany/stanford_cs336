import tokenizer.bpe_encoding as bpe_encoding
import tokenizer_training_encoding.dataloader as dataloader
import tokenizer.bpe as bpe
import numpy as np
import time


if __name__=='__main__':
	# filename = 'TinyStoriesV2-GPT4-train'
	filename = 'owt_train'
	input_path = f'./data/{filename}.txt'
	output_path = f'./data/{filename}_tokenized.txt' 
	# vocab_filepath = 'train_bpe_tinystories_vocab.txt'
	# merges_filepath = 'train_bpe_tinystories_merges.txt'
	vocab_filepath = 'train_bpe_owt_train_vocab.txt'
	merges_filepath = 'train_bpe_owt_train_merges.txt'
	special_tokens=[bpe.END_OF_TEXT]

	start = time.time()

	loader = dataloader.DataLoader(input_path=input_path, special_tokens=special_tokens)
	chunks = loader.load_data_chunk(num_chunks=100)
	tokenizer = bpe_encoding.Tokenizer.from_files(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=special_tokens)

	results = []

	for i, chunk in enumerate(chunks):
		print(f'Processing chunk {i}')
		encoded_documents = tokenizer.parallel_encode(chunk)
		serialized_documents = [np.array(document, dtype=np.uint16) for document in encoded_documents]
		results.extend(serialized_documents)
		results.append(np.array([tokenizer.endoftext_token]))

	result = np.concatenate(results)

	np.save(output_path, result)
	print(f'Saved output to: {output_path}')
	print(f'Time elapsed: {time.time() - start}')
