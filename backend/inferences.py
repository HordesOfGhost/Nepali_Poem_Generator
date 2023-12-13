from backend.transformer import *

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
tf.config.run_functions_eagerly(True) 


def inference_from_transformer(prompt_text, model_name, tokens_to_generate):
    model = AutoregressiveWrapper
    tr = model.load_checkpoint(f'backend/models/Transformers/{model_name}')

    tokenizer = NepaliTokenizer()

    generator = Generator(tr, tokenizer)

    generated_text = generator.generate(
        max_tokens_to_generate = tokens_to_generate,
        prompt = prompt_text,
        padding_token = tokenizer.character_to_token('<pad>')
    )
    return generated_text.replace('<pad>', '')

def inference_from_nepali_lstm(prompt_text, tokens_to_generate):
    
    # Load the tokenizer from the JSON file
    with open('backend/models/LSTM/tokenizer.json', 'r', encoding='utf-8') as json_file:
        tokenizer_json = json_file.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
    
    # Load model
    model = load_model('backend/models/LSTM/LSTM.h5')
    max_sequence_len = 11

    for _ in range(tokens_to_generate):
        token_list = tokenizer.texts_to_sequences([prompt_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1,
        padding='pre')
        predicted = np.argmax(model.predict(token_list,
                                            verbose=0), axis=-1)
        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        prompt_text += " " + output_word

    return prompt_text

def inference_from_distil_gpt2(prompt_text, token_size):
    
    model_path = "backend/models/DistillGPT-2/DistillGPT2"

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    ids = tokenizer.encode(f'{prompt_text}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=token_size,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return generated_text

def inference_from_gpt2(prompt_text, token_size):
    
    model_path = "backend/models/GPT-2/GPT2"

    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    ids = tokenizer.encode(f'{prompt_text}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=token_size,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return generated_text