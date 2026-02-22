import tokenizer.bpe as bpe
import time
import json
import base64
import tracemalloc

if __name__ == '__main__':
	tracemalloc.start()
	start = time.time()

	input_path = './data/TinyStoriesV2-GPT4-train.txt'
	vocab_size = 10000
	special_tokens = [bpe.END_OF_TEXT]
	tokenizer = bpe.BytePairEncoderTokenizer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
	vocab, merges = tokenizer.train()

	vocab_serializable = {str(k): base64.b64encode(v).decode('ascii') for k, v in vocab.items()}

	current, peak = tracemalloc.get_traced_memory()
	print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
	print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

	with open('train_bpe_tinystories_vocab.txt', 'w', encoding='utf-8') as f:
		json.dump(vocab_serializable, f, indent=4)

	with open('train_bpe_tinystories_merges.txt', 'w', encoding='utf-8') as f:
		for merge in merges:
			merge_readable = ' + '.join(base64.b64encode(b).decode('ascii') for b in merge)
			f.write(f'{merge_readable}\n')

	tracemalloc.stop()
	print('Time elapsed: ', time.time() - start)