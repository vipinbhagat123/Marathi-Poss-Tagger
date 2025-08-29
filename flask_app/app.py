from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load the BiLSTM model
model = tf.keras.models.load_model('model_bilstm.h5')

# Load tokenizers
with open('word_tokenizer.pkl', 'rb') as f:
    word_tokenizer = pickle.load(f)

with open('tag_tokenizer.pkl', 'rb') as f:
    tag_tokenizer = pickle.load(f)

MAX_LEN = 84  # Adjust this as per your model training

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        sentence = request.form['sentence']
        words = sentence.strip().split()

        # Convert words to sequences
        seq = word_tokenizer.texts_to_sequences([words])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        predictions = model.predict(padded)
        pred_tags = np.argmax(predictions, axis=-1)

        # Decode tags
        index_tag = {v: k for k, v in tag_tokenizer.word_index.items()}
        index_tag[0] = 'PAD'

        tags = [index_tag.get(i, 'UNK') for i in pred_tags[0][:len(words)]]
        result = list(zip(words, tags))

    return render_template('index.html', result=result)

@app.route('/info')
def info():
    model_info = {
        'Simple RNN':  {'Train Acc': '85.2%', 'Test Acc': '81.6%', 'Train Loss': '0.45', 'Test Loss': '0.51'},
        'LSTM':        {'Train Acc': '90.4%', 'Test Acc': '87.3%', 'Train Loss': '0.31', 'Test Loss': '0.38'},
        'BiLSTM':      {'Train Acc': '93.1%', 'Test Acc': '89.7%', 'Train Loss': '0.24', 'Test Loss': '0.33'},
        'GRU':         {'Train Acc': '91.2%', 'Test Acc': '88.1%', 'Train Loss': '0.28', 'Test Loss': '0.36'},
    }

    team_info = {
        'Team Name': 'POS Mavericks',
        'Members': ['Prajakta S.', 'Member 2', 'Member 3']
    }

    return render_template('info.html', model_info=model_info, team_info=team_info)

if __name__ == '__main__':
    app.run(debug=True, port = 5001)
