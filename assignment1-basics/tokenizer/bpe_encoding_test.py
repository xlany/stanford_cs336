from tokenizer.bpe_encoding import Tokenizer

def test_simple():
	text = 'the cat ate'
	vocab = {0: b' ',
			1: b'a',
			2: b'c',
			3: b'e',
			4: b'h',
			5: b't',
			6: b'th',
			7: b' c',
			8: b' a',
			9: b'the',
			10: b' at'}

	merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

	tokenizer = Tokenizer(vocab, merges)
	assert tokenizer.encode(text) == [9, 7, 1, 5, 10, 3]

def test_special_tokens():
	text = 'the cat ate<|endoftext|>'
	vocab = {0: b' ',
			1: b'a',
			2: b'c',
			3: b'e',
			4: b'h',
			5: b't',
			6: b'th',
			7: b' c',
			8: b' a',
			9: b'the',
			10: b' at',
			11: b'<|endoftext|>'}

	merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
	end_of_text = '<|endoftext|>'

	tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[end_of_text])
	assert tokenizer.encode(text) == [9, 7, 1, 5, 10, 3, 11]

def test_special_tokens_multiple():
	text = 'the cat ate<|endoftext|><|endoftext|>'
	vocab = {0: b' ',
			1: b'a',
			2: b'c',
			3: b'e',
			4: b'h',
			5: b't',
			6: b'th',
			7: b' c',
			8: b' a',
			9: b'the',
			10: b' at',
			11: b'<|endoftext|>'}

	merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
	end_of_text = '<|endoftext|>'

	tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[end_of_text])
	assert tokenizer.encode(text) == [9, 7, 1, 5, 10, 3, 11, 11]

def test_encode_decode():
	text = 'the cat ate'
	vocab = {0: b' ',
			1: b'a',
			2: b'c',
			3: b'e',
			4: b'h',
			5: b't',
			6: b'th',
			7: b' c',
			8: b' a',
			9: b'the',
			10: b' at'}

	merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

	tokenizer = Tokenizer(vocab, merges)
	encoded = tokenizer.encode(text)
	assert encoded == [9, 7, 1, 5, 10, 3]
	decoded = tokenizer.decode(encoded)
	assert decoded == text


def test_encode_decode_special_tokens():
	text = 'the cat ate<|endoftext|>'
	vocab = {0: b' ',
			1: b'a',
			2: b'c',
			3: b'e',
			4: b'h',
			5: b't',
			6: b'th',
			7: b' c',
			8: b' a',
			9: b'the',
			10: b' at',
			11: b'<|endoftext|>'}

	merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
	end_of_text = '<|endoftext|>'

	tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[end_of_text])
	encoded = tokenizer.encode(text)
	assert encoded == [9, 7, 1, 5, 10, 3, 11]
	decoded = tokenizer.decode(encoded)
	assert decoded == text

def test_decode_unknown():
	expected = 'the cat ate\ufffd'
	vocab = {0: b' ',
			1: b'a',
			2: b'c',
			3: b'e',
			4: b'h',
			5: b't',
			6: b'th',
			7: b' c',
			8: b' a',
			9: b'the',
			10: b' at'}

	merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

	tokenizer = Tokenizer(vocab, merges)
	encoded = [9, 7, 1, 5, 10, 3, 60]
	decoded = tokenizer.decode(encoded)
	assert decoded == expected
