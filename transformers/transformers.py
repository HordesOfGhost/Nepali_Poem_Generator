import torch
from models import *

poem_file = open('C:/Users/Ghost/Desktop/gits/Nepali_Poem_Generator/datasets/poem.txt','r',encoding='utf-8')
poem = poem_file.read()


poem_corpus = poem.split("\n")
processed_poem_corpus = remove_noise(poem_corpus)



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


tokenizer = NepaliTokenizer()

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
))

# Create the training data
training_data = '। '.join(processed_poem_corpus)

tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

# Train the model
print("Training")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(model, tokenizer, optimizer)
trainer.train(sequences, epochs=100, batch_size=8)

# Save
def save_checkpoint(self, path):
    print(f'Saving checkpoint {path}')
    torch.save({
        'number_of_tokens': self.number_of_tokens,
        'max_sequence_length': self.max_sequence_length,
        'embedding_dimension': self.embedding_dimension,
        'number_of_layers': self.number_of_layers,
        'number_of_heads': self.number_of_heads,
        'feed_forward_dimension': self.feed_forward_dimension,
        'dropout_rate': self.dropout_rate,
        'model_state_dict': self.state_dict()
    }, path)

@staticmethod
def load_checkpoint(path) -> 'LanguageModel':
    checkpoint = torch.load(path)
    model = LanguageModel(
        number_of_tokens=checkpoint['number_of_tokens'],
        max_sequence_length=checkpoint['max_sequence_length'],
        embedding_dimension=checkpoint['embedding_dimension'],
        number_of_layers=checkpoint['number_of_layers'],
        number_of_heads=checkpoint['number_of_heads'],
        feed_forward_dimension=checkpoint['feed_forward_dimension'],
        dropout_rate=checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_checkpoint(self, path):
    self.model.save_checkpoint(path)

@staticmethod
def load_checkpoint(path) -> 'AutoregressiveWrapper':
    model = LanguageModel.load_checkpoint(path)
    return AutoregressiveWrapper(model)

model.save_checkpoint('./trained_model')
model = model.load_checkpoint('./trained_model')