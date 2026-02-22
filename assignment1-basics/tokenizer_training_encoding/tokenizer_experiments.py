import tokenizer.bpe_encoding as bpe_encoding
import tokenizer.bpe as bpe
import os
import tokenizer_training_encoding.dataloader as dataloader


if __name__=='__main__':
	# input_path = './data/TinyStoriesV2-GPT4-train.txt'
	# vocab_filepath = 'train_bpe_tinystories_vocab.txt'
	# merges_filepath = 'train_bpe_tinystories_merges.txt'

	tinystories_input_path = './data/TinyStoriesV2-GPT4-train.txt'
	tinystories_vocab_filepath = 'train_bpe_tinystories_vocab.txt'
	tinystories_merges_filepath = 'train_bpe_tinystories_merges.txt'

	input_path = './data/owt_valid.txt'
	vocab_filepath = 'train_bpe_owt_train_vocab.txt'
	merges_filepath = 'train_bpe_owt_train_merges.txt'

	owt_input_path = './data/owt_valid.txt'
	owt_vocab_filepath = 'train_bpe_owt_train_vocab.txt'
	owt_merges_filepath = 'train_bpe_owt_train_merges.txt'

	special_tokens=[bpe.END_OF_TEXT]
	samples = 10

	loader = dataloader.DataLoader(input_path=input_path, special_tokens=special_tokens)
	random_documents = loader.sample_random_documents(k=samples)
	# tokenizer = bpe_encoding.Tokenizer.from_files(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath, special_tokens=special_tokens)

	tinystories_tokenizer = bpe_encoding.Tokenizer.from_files(vocab_filepath=tinystories_vocab_filepath, merges_filepath=tinystories_merges_filepath, special_tokens=special_tokens)
	owt_tokenizer = bpe_encoding.Tokenizer.from_files(vocab_filepath=owt_vocab_filepath, merges_filepath=owt_merges_filepath, special_tokens=special_tokens)


	# for document in random_documents:
	# 	encoded_document = tokenizer.encode(document)
	# 	original_length = len(document.encode("utf-8"))
	# 	encoded_length = len(encoded_document)
	# 	print(f'Original length in bytes: {original_length}')
	# 	print(f'Encoded length in bytes: {encoded_length}')
	# 	print(f'Ratio: {round(original_length/encoded_length, 2)}')

	for document in random_documents:
		tinystories_encoded_document = tinystories_tokenizer.encode(document)
		owt_encoded_document = owt_tokenizer.encode(document)

		original_length = len(document.encode("utf-8"))
		tinystories_encoded_length = len(tinystories_encoded_document)
		owt_encoded_length = len(owt_encoded_document)
		print(f'Original length in bytes: {original_length}')
		# print(f'Encoded length in bytes: {encoded_length}')
		print(f'Ratio TinyStories: {round(original_length/tinystories_encoded_length, 2)}')
		print(f'Ratio OpenWebText: {round(original_length/owt_encoded_length, 2)}')


