from tokenizer_training_encoding.dataloader import DataLoader

def test_data_loader():
  loader = DataLoader(input_path='./data/simple.txt', special_tokens=['<|endoftext|>'])
  data = loader.load_data_chunk(num_chunks=1)
  assert data == [['story1<|endoftext|>', 'story2<|endoftext|>']]