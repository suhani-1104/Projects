from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load LSTM model and tokenizer
model = load_model('spam_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # Use the same maxlen as used during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Convert text to padded sequence
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    # Predict with the LSTM model
    prediction = model.predict(padded)[0][0]
    result = 'SPAM' if prediction < 0.5 else 'HAM'

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
