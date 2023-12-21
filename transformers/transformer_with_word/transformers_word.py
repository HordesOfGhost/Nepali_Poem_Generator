import torch
from models_word import *

poem_file = open('datasets/poem.txt','r',encoding='utf-8')
poem = poem_file.read()

poem_corpus = poem.split("\n")
processed_poem_corpus = remove_noise(poem_corpus)
processed_poem_corpus_text = ' '.join(processed_poem_corpus)

def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences


def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    tokenized_training_data = tokenizer.tokenize(training_data)
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))
    return tokenized_training_data

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


tokenizer = NepaliTokenizer(processed_poem_corpus_text)

embedding_dimension = 256
max_sequence_length = 20
number_of_tokens = tokenizer.size()

# Create the model
model = AutoregressiveWrapper(LanguageModel(
    embedding_dimension=embedding_dimension,
    number_of_tokens=number_of_tokens,
    number_of_heads=4,
    number_of_layers=3,
    dropout_rate=0.1,
    max_sequence_length=max_sequence_length
)).to(get_device())

# Create the training data
training_data = '। '.join(processed_poem_corpus)

tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

# Train the model
print(f"Training on {get_device()}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(model, tokenizer, optimizer)
trainer.train(sequences, epochs=250, batch_size=8)


model.save_checkpoint('./trained_model')
model = model.load_checkpoint('./trained_model')