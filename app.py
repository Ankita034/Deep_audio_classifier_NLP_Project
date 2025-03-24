import os
from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model and label encoder
model = load_model('model.h5')
labelencoder = joblib.load('labelencoder.joblib')

# Set the path to the uploaded files directory
UPLOAD_FOLDER = 'UrbanSound8K'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_sound(filename):
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    return prediction_class[0]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict the sound
            prediction = predict_sound(file_path)
    
    return render_template('upload.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
