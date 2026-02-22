import os
import collections
from typing import Any
import pytest
import tokenizer.bpe as bpe

def test_simple():
    input_path = './data/test.txt'
    vocab_size = 263
    special_tokens = [bpe.END_OF_TEXT]
    tokenizer = bpe.BytePairEncoderTokenizer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    vocab, merges = tokenizer.train()
    assert tokenizer.new_words() == [b'st', b'est', b'ow', b'low', b'west', b'ne']
    assert merges == [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
    assert len(vocab) == vocab_size

