from models import *

model = AutoregressiveWrapper
tr = model.load_checkpoint('./trained_model')

poem_file = open('datasets/poem.txt','r',encoding='utf-8')
poem = poem_file.read()
poem_corpus = poem.split("\n")
processed_poem_corpus = remove_noise(poem_corpus)
processed_poem_corpus_text = ' '.join(processed_poem_corpus)

tokenizer = NepaliTokenizer(processed_poem_corpus_text)


max_tokens_to_generate = 50
generator = Generator(tr, tokenizer)
generated_text = generator.generate(
    max_tokens_to_generate=max_tokens_to_generate,
    prompt="यो मेरो",
    padding_token=tokenizer.character_to_token('<pad>')
)

generated_texts = generated_text.replace('<pad>', '')
with open('txtt.txt', 'w', encoding='utf-8') as file:
    file.write(generated_texts)
