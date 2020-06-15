from tashaphyne.stemming import ArabicLightStemmer
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
ArListem = ArabicLightStemmer()

classes = ['Calture','Diverse','Economy','Politics','Sport']

def text_to_tfidf(input_text):
    tfidf_model = pickle.load(open('model/tfidf_model.sav', 'rb'))
    temp_text = [] 
    for word in input_text.split(" "):
        stem = ArListem.light_stem(word)
        stem = ArListem.get_stem()
        temp_text.append(stem)
    temp_text = pd.DataFrame([" ".join(temp_text)])
    tfidf_text = tfidf_model.transform(temp_text[0])
    return tfidf_text

def classification_nb(input_text):
    nb_model = pickle.load(open('model/naive_bayes_model.sav', 'rb'))
    tfidf_text = text_to_tfidf(input_text)
    label = nb_model.predict(tfidf_text)[0]
    return classes[label]

def classification_svm(input_text):
    svm_model = pickle.load(open('model/svm_model.sav', 'rb'))
    tfidf_text = text_to_tfidf(input_text)
    label = svm_model.predict(tfidf_text)[0]
    return classes[label]

def text_to_df(input_text):
    temp_text = [] 
    for word in input_text.split(" "):
        stem = ArListem.light_stem(word)
        stem = ArListem.get_stem()
        temp_text.append(stem)
    temp_text = pd.DataFrame([" ".join(temp_text)])
    return temp_text[0]

def classification_cnn(input_text):
    tok = pickle.load(open('model/tokenizer_cnn.pickle', 'rb'))
    cnn_model = load_model('model/cnn_model.h5')
    temp_text = text_to_df(input_text)
    test_sequences = tok.texts_to_sequences(temp_text)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=150)
    label = np.argmax(cnn_model.predict([[test_sequences_matrix[0]]],batch_size=1,verbose = 2)[0])
    return classes[label]

def classification_lstm(input_text):
    tok = pickle.load(open('model/tokenizer_lstm.pickle', 'rb'))
    lstm_model = load_model('model/lstm_model.h5')
    temp_text = text_to_df(input_text)
    test_sequences = tok.texts_to_sequences(temp_text)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=150)
    label = np.argmax(lstm_model.predict([[test_sequences_matrix[0]]],batch_size=1,verbose = 2)[0])
    return classes[label]

