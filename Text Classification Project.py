# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import json
import os

from modules import text_cleaning, lstm_model_creation

# %% Functions

# %% Step 1) Data Loading
CSV_PATH = os.path.join(os.getcwd(), 'Dataset', 'True.csv')
df = pd.read_csv(CSV_PATH)

# %% Step 2) Data Inspection
df.info()
df.describe()
df.duplicated().sum()  # 206 Duplicated

# %% Step 3) Data Cleaning
# Things to be removed
for index, temp in enumerate(df['text']):
    df['text'][index] = text_cleaning(temp)

    # # combined regex pattern
    # out = re.sub('bit.ly/\d\w{1,10}|@[^\s]+|^.*?\)\s*-|\[.*?EST\]|[^a-zA-Z]',' ',temp)
    # print(out)

# %% Step 4) Features Selection
X = df['text']
y = df['subject']

# %% Step 5) Data Preprocessing
# Tokenizer
# need to identify via checking the unique words in the setences
num_words = 5000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# padding
X = pad_sequences(X, maxlen=300, padding='post', truncating='post')
# %%
# OHE
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y[::, None])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=123, train_size=0.2)

# %% Model Development
model = lstm_model_creation(num_words, y.shape[1], dropout=0.4)
hist = model.fit(X_train, y_train, epochs=5, batch_size=64,
                 validation_data=(X_test, y_test))

# %% Model analysis
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

# %% Model Saving
# save trained tf model
model.save('model.h5')

# save ohe
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)
