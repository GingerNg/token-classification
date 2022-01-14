from models.base import NerModel
import models.cck_process_data as process_data
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from models.CRF import MyCRF
from tensorflow.keras import layers

def BilstmCRFKerasModel(EMBED_DIM, vocab_size, n_tags, BiRNN_UNITS=100, use_crf=True, use_mask=True):
    # function API
    s_input = layers.Input(shape=(None,))
    emb_layer = layers.Embedding(vocab_size,
                                 EMBED_DIM,
                                 embeddings_initializer=keras.initializers.Orthogonal(
                                     gain=1.0, seed=1),
                                 trainable=True,
                                 mask_zero=use_mask,
                                 #   embeddings_initializer=initializer
                                 )
    lstm_layer1 = layers.Bidirectional(layers.LSTM(BiRNN_UNITS // 2,
                                                   return_sequences=True,
                                                   dropout=0.1,
                                                   recurrent_dropout=0.1))
    lstm_layer2 = layers.Bidirectional(layers.LSTM(BiRNN_UNITS // 2,
                                                   return_sequences=True,
                                                   dropout=0.1,
                                                   recurrent_dropout=0.1))
    crf = MyCRF(n_tags, name="crf_layer")
    output = crf(
        lstm_layer2(lstm_layer1(emb_layer(s_input))))
    model = keras.Model(s_input, output)
    model.compile('adam',
                  run_eagerly=True,   # 使用动态图
                  loss={'crf_layer': crf.crf_loss},
                  metrics=[crf.crf_accuracy]
                  )
    return model


class BilstmCRFKerasNerModel(NerModel):
    def __init__(self, chunk_tags, vocab_size=1000, emb_dim=100, model_path=None, emb_use_pretrained=0) -> None:
        super().__init__()
        use_crf = True
        n_tags = len(chunk_tags)
        self.chunk_tags = chunk_tags
        self.model = BilstmCRFKerasModel(emb_dim, vocab_size, n_tags, use_crf=use_crf)
        self.model.summary()
        self.model_path = model_path
        self.maxlen = 36
        self.encoder = None


    def fit(self,train_data, val_data, epochs=10, batch_size=32):
        train_x, train_y = train_data
        val_x, val_y = val_data
        histroy = self.model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            # validation_data=(test_x, test_y),
                            validation_data=(val_x, val_y),
                            callbacks=[ModelCheckpoint(self.model_path, # '{}/crf_{}.h5'.format(ner_model.model_dir, dataset_name)
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True,
                                    save_weights_only=True),
                                        EarlyStopping(
                            monitor='val_loss', patience=3)]
                            )


    def preprocess(self, data, onehot=False):
        if self.maxlen is None:
            self.maxlen = max(len(s) for s in data)
        x = [self.encoder.encode_ids(s) for s in data]  # set to <unk> (index 1) if not in vocab
        y_chunk = [[self.chunk_tags.index(w[1]) for w in s] for s in data]
        x = pad_sequences(x, self.maxlen, value=self.encoder.word2idx["pad"], padding='post')
        y_chunk = pad_sequences(y_chunk, self.maxlen, value=self.chunk_tags.index("O"), padding='post')
        print("y_chunk shape:", y_chunk.shape)

        if onehot:
            y_chunk = np.eye(len(self.chunk_tags), dtype='float32')[y_chunk]
        else:
            pass
            # y_chunk = numpy.expand_dims(y_chunk, 2)
        return x, y_chunk


    def load_model(self):
        pass

    def predict(self):
        pass

    def batch_predict(self):
        pass

    def evaluate(self):
        pass