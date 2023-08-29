import nltk 
import tensorflow as tf

import numpy as np
import json
import string
import sklearn
import pickle


def preprocess(text):
    def remove_punc(text):
        text_nopunc = ''.join([letter for letter in text.lower() if letter not in string.punctuation])
        return text_nopunc

    def remove_stopwords(text):
        text_stopwords_removed = ' '.join([word for word in nltk.word_tokenize(text) if word not in nltk.corpus.stopwords.words('english')])
        return text_stopwords_removed
    
    lemmatizer = nltk.WordNetLemmatizer()
    return (remove_stopwords(lemmatizer.lemmatize(remove_punc(text))))


def predict(text, tokenizer, label_tokenizer):
    sequence = tokenizer.texts_to_sequences([preprocess(text)])
    #padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding = 'post', maxlen=MAX_WORDS)
    print(sequence)
    model = tf.keras.models.load_model('zoe_model.h5')
    prediction = model.predict(sequence)
    print(np.argmax(prediction))
    disease = label_tokenizer.inverse_transform([np.argmax(prediction)])[0]
    return disease


if __name__ == '__main__':
    MAX_WORDS = 10
    EMBEDDING_DIM = 32
    tags = []
    patterns = []
    with open('intents.json', 'r') as file:
        data = json.load(file)['intents']
        for case in data:
            for case_pattern in case['patterns']:
                patterns.append(case_pattern)
                tags.append(case['tag'])

    print(tags)


    patterns_stop_rem = np.array(list(map(preprocess,patterns)))


    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(patterns_stop_rem)

    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(patterns_stop_rem)

    label_tokenizer = sklearn.preprocessing.LabelEncoder()
    labels_tokens = label_tokenizer.fit_transform(tags)
    print(labels_tokens)


    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding = 'post', maxlen=MAX_WORDS)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word_index) + 1, EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(tags), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(x = padded_sequences, y = labels_tokens, epochs = 200)

    model.save('zoe_model.h5')
    with open('tokenizer.pickle', 'wb') as encoder:
        pickle.dump(tokenizer, encoder)
    with open('label_tokenizer.pickle', 'wb') as label_encoder:
        pickle.dump(label_tokenizer,label_encoder)
    

