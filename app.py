from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import os
import logging

# 🔧 Initialize Flask app
app = Flask(__name__)

# 📁 Folder to store uploaded images
target_img = 'static/uploads'
app.config['UPLOAD_FOLDER'] = target_img

# 🔐 Ensure the upload directory exists
os.makedirs(target_img, exist_ok=True)

# 🧠 Load the trained model
model = load_model('code/vgg16_model.h5')  # Update to 'inception_model.h5' or 'xception_model.h5' as needed

# 🏷️ Class label dictionary
card_names = {
    0:  'ace of clubs',
    1:  'ace of diamonds',
    2:  'ace of hearts',
    3:  'ace of spades',
    4:  'eight of clubs',
    5:  'eight of diamonds',
    6:  'eight of hearts',
    7:  'eight of spades',
    8:  'five of clubs',
    9:  'five of diamonds',
    10: 'five of hearts',
    11: 'five of spades',
    12: 'four of clubs',
    13: 'four of diamonds',
    14: 'four of hearts',
    15: 'four of spades',
    16: 'jack of clubs',
    17: 'jack of diamonds',
    18: 'jack of hearts',
    19: 'jack of spades',
    20: 'joker',
    21: 'king of clubs',
    22: 'king of diamonds',
    23: 'king of hearts',
    24: 'king of spades',
    25: 'nine of clubs',
    26: 'nine of diamonds',
    27: 'nine of hearts',
    28: 'nine of spades',
    29: 'queen of clubs',
    30: 'queen of diamonds',
    31: 'queen of hearts',
    32: 'queen of spades',
    33: 'seven of clubs',
    34: 'seven of diamonds',
    35: 'seven of hearts',
    36: 'seven of spades',
    37: 'six of clubs',
    38: 'six of diamonds',
    39: 'six of hearts',
    40: 'six of spades',
    41: 'ten of clubs',
    42: 'ten of diamonds',
    43: 'ten of hearts',
    44: 'ten of spades',
    45: 'three of clubs',
    46: 'three of diamonds',
    47: 'three of hearts',
    48: 'three of spades',
    49: 'two of clubs',
    50: 'two of diamonds',
    51: 'two of hearts',
    52: 'two of spades'
}

# 📜 Logging setup
logging.basicConfig(level=logging.DEBUG)

# 📍 Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# 📍 Route: Upload Page
@app.route('/input')
def input_page():
    return render_template('input.html')

# 📍 Route: Prediction
@app.route('/predict', methods=['POST'])
def output():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400

        # ✅ Secure and save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Saved file: {filepath}")

        # 🖼️ Preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Update to (299, 299) if needed
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 🔮 Predict
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        card_name = card_names.get(predicted_index, 'Unknown Card')

        # 🎯 Return result on output.html
        return render_template('output.html',
                               prediction=card_name,
                               image_path=f'static/uploads/{filename}')

    except Exception as e:
        logging.exception("Prediction error:")
        return jsonify({'error': str(e)}), 500

# 🚀 Main app launcher
if __name__ == '__main__':
    if not os.path.exists(target_img):
        os.makedirs(target_img)
    app.run(debug=True)