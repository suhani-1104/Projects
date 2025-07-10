from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

model = load_model('spam_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]

    print("Message:", message)
    print("Tokenized:", seq)
    print("Padded:", padded)
    print("Raw prediction:", prediction)

    result = 'SPAM' if prediction > 0.15 else 'HAM'
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
