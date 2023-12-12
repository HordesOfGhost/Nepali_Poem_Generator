from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling
import math

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(sequence, max_length):
    model_path = "dGPT2"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    print(generated_text)
        # Save the generated text to a file
    with open('txtt.txt', 'w', encoding='utf-8') as file:
        file.write(generated_text)

sequence = input()
max_len = int(input()) # 20
generate_text(sequence, max_len)