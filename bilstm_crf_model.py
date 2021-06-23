from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from CRF import CRF
import process_data
import pickle
from tensorflow.keras.layers import Input

EMBED_DIM = 200
BiRNN_UNITS = 200
model_dir = "saved_model"


def create_model(train=True, maxlen=100):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab,
                                               chunk_tags) = process_data.load_data()
    else:
        with open('{}/config.pkl'.format(model_dir), 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    n_tags = len(chunk_tags)

    n_tags = len(chunk_tags)
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(n_tags, name="crf_layer")
    model.add(crf)
    model.compile('adam', loss={'crf_layer': crf.crf_loss},
                  metrics=[crf.crf_accuracy])
    model.summary()

    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
