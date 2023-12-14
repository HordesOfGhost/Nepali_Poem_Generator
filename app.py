from flask import Flask, render_template, request
from backend.inferences import *

app = Flask(__name__)

def split_lines(text, punctuation_marks):
    for mark in punctuation_marks:
        text = text.replace(mark, f'{mark}flag')
    text_line = text.split('flag') 
    return text_line

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission here
        prompt_text = request.form.get('text_area')
        selected_model = request.form.get('dropdown')
        token_size = int(request.form.get('token_size'))
        
        if selected_model == 'TransformerWtChar' or selected_model == 'TransformerWtWord':
            generated_text = inference_from_transformer(prompt_text, selected_model, token_size)

        elif selected_model == 'LSTM':
            generated_text = inference_from_nepali_lstm(prompt_text, token_size)
        
        elif selected_model == 'DistilGPT2':
            generated_text = inference_from_distil_gpt2(prompt_text, token_size)

        elif selected_model == 'GPT2':
            generated_text = inference_from_distil_gpt2(prompt_text, token_size)
            
        else:
            generated_text = ''

        punctuations = ['!।','! ।','!|',',।','|','!', '.', '।' , '?']
        generated_text_line = split_lines(generated_text, punctuations)

        # Render the template with the form
        return render_template('index.html',generated_text_lines = generated_text_line, selected_model = selected_model, prompt_text = prompt_text, token_to_generate = token_size )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
