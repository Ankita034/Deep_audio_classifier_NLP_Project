# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")
# -

# !pip install librosa

import sys
print(sys.executable)


# !python -c "import librosa; print(librosa.__version__)"


import IPython.display as ipd
import librosa
import librosa.display

filename = 'UrbanSound8K/fold5/190893-2-0-11.wav'


# Librosa

#

librosa_audio_data,librosa_sample_rate = librosa.load(filename)


librosa_audio_data


librosa_sample_rate


librosa.display.waveshow(librosa_audio_data,sr=librosa_sample_rate)
ipd.Audio(filename)

# Here Librosa converts the signal to mono, meaning the channel will alays be 1



# scipy

#

from scipy.io import wavfile as wav
wave_sample_rate,wave_audio = wav.read(filename)

wave_audio

wave_sample_rate


plt.figure(figsize=(12,4))
plt.plot(wave_audio)
plt.show()

# Here Scipy converts the signal to stereo, meaning the channels will be 2¶
#



# Metadata



metadata = pd.read_csv('UrbanSound8K.csv')
metadata.head()

metadata.shape


metadata.isnull().sum()


metadata.duplicated().sum()


#check whether the dataset is balanced/imbalanced
metadata['class'].value_counts()

# The data is balanced
#



# Data Visualization



sns.countplot(x='class',data=metadata)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15,8))
plt.title('Audio Class Distribution')
class_dist = metadata['class'].value_counts().sort_values()
sns.barplot(x=class_dist.values,
           y=class_dist.keys())
plt.show()

plt.figure(figsize=(15,8))
plt.title('Audio Folds Distribution')
folds_dist = metadata['fold'].value_counts().sort_values()
sns.barplot(y=folds_dist.values,
           x=folds_dist.keys())
plt.show()

#

# Data Preprocessing



# Feature Extraction
#
#
#
# Here we will be using Mel-Frequency Cepstral Coefficients(MFCC) from the audio samples.
# The MFCC summarises the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound.
# These audio representations will allow us to identify features for classification.

mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=50)
mfccs

mfccs.shape


# This is the preprocessing for a single individual file

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name) 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


audio_dataset_path='UrbanSound8K'


# !pip install tqdm


import os
from tqdm import tqdm

### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for i,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


#converting extracted features to pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()

# Data Splitting

#

# +

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# -

X

X.shape

y.shape

# Categorical Encoding

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

# !pip install tensorflow

# Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)

X_train.shape

X_test.shape


y_train.shape

y_test.shape

print("Number of training samples = ", X_train.shape[0])
print("Number of testing samples = ",X_test.shape[0])



# Model Building

#

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,classification_report

#No of classes
num_labels = y.shape[1]
num_labels

# +
model = Sequential()
# first layer
model.add(Dense(256,input_shape=(50,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# second layer
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#third layer
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))

#final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
# -

model.summary()

# !pip install pydot

from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)

from tensorflow.keras.utils import to_categorical


model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam')



# Model Training
#

# +
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 150
num_batch_size = 32
# -

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)


training = model.fit(X_train,
                     y_train,
                     batch_size=num_batch_size,
                     epochs=num_epochs,
                    validation_data=(X_test,y_test))

test_accuracy = model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

train_hist = pd.DataFrame(training.history)
train_hist

# +
# Loss curves
plt.figure(figsize=[12,4])
plt.plot(training.history['loss'],'red',linewidth=3.0)
plt.plot(training.history['val_loss'],'orange',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=15)
plt.xlabel('Epochs ',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.title('Loss Curves',fontsize=15)


# Accuracy Curves
plt.figure(figsize=[12,4])
plt.plot(training.history['accuracy'],'darkgreen',linewidth=3.0)
plt.plot(training.history['val_accuracy'],'lightgreen',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=15)
plt.xlabel('Epochs ',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title('Accuracy Curves',fontsize=15)
plt.show()

# -

ytrue = np.argmax(y_test,axis=1)
ypred = np.argmax(model.predict(X_test),axis=1)

from sklearn.metrics import confusion_matrix, classification_report
print('\nConfusion Matrix :\n\n')
print(confusion_matrix(ytrue,ypred))

# +
plt.figure(figsize=(10,4))
plt.title("Confusion matrix for testing data", fontsize = 15)
plt.xlabel("Predicted class")
plt.ylabel("True class")
sns.heatmap(confusion_matrix(ytrue,ypred),annot=True,
           xticklabels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'enginge_idling', 'gun_shot', 'jackhammer', 'siren','street_music'],
           yticklabels=['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'enginge_idling', 'gun_shot', 'jackhammer', 'siren','street_music'])

plt.show()
# -

print('\n\nClassification Report : \n\n',classification_report(ytrue,ypred))



# Model Prediction

X_test[1]

filename = 'UrbanSound8K/fold5/190893-2-0-11.wav'
prediction_feature = features_extractor(filename)
prediction_feature = prediction_feature.reshape(1,-1)
np.argmax(model.predict(prediction_feature),axis=1)

prediction_feature.shape

np.argmax(model.predict(X_test),axis=1)

#

# Testing Some Test Audio Data



filename = "UrbanSound8K/fold8/103076-3-0-0.wav"
audio,sample_rate = librosa.load(filename)
mfccs_features = librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=50)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

mfccs_scaled_features


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
mfccs_scaled_features

mfccs_scaled_features.shape


predicted_label = np.argmax(model.predict(mfccs_scaled_features),axis=1)
predicted_label

prediction_class = labelencoder.inverse_transform(predicted_label)
prediction_class

ipd.Audio(filename)


# Predictions
#

def predict(filename):
    audio, sample_rate = librosa.load(filename) 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label = np.argmax(model.predict(mfccs_scaled_features),axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    print(prediction_class)
    return ipd.Audio(filename)



predict('UrbanSound8K/fold8/99179-9-0-38.wav')


predict('UrbanSound8K/fold9/12812-5-0-0.wav')


predict('UrbanSound8K/fold9/196066-2-0-1.wav')





# SAVING THE MODEL

model.save('model.h5')


import joblib
joblib.dump(labelencoder, 'labelencoder.joblib')


