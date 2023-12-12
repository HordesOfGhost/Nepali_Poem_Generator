from models import *

model = AutoregressiveWrapper
tr = model.load_checkpoint('./trained_model')

tokenizer = NepaliTokenizer()


max_tokens_to_generate = 50
generator = Generator(tr, tokenizer)
generated_text = generator.generate(
    max_tokens_to_generate=max_tokens_to_generate,
    prompt="ननिभ्ने",
    padding_token=tokenizer.character_to_token('<pad>')
)
print(generated_text.replace('<pad>', ''))