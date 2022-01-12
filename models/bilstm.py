from tensorflow import keras
# from tensorflow.keras import models
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, BatchNormalization, Dropout
from tensorflow_addons.layers import CRF
# from models.CRF import ModelWithCRFLoss
from models.CRF import crf_loss, MyCRF
from tensorflow.keras import layers


EMBED_DIM = 100
BiRNN_UNITS = 100
model_dir = "saved_model"


def create_model(EMBED_DIM, vocab, n_tags, use_crf=True, use_mask=True):
    # function API
    s_input = layers.Input(shape=(None,))
    emb_layer = layers.Embedding(len(vocab),
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
    # print(type(output))
    model = keras.Model(s_input, output)
    # model = ModelWithCRFLoss(base_model)

    # model = Sequential()
    # model.add(Embedding(len(vocab),
    #                     EMBED_DIM,
    #                     embeddings_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
    #                     trainable=True,
    #                     mask_zero=True
    #                     )
    #           )  # Random embedding
    # model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.8))
    # model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    # # model.add(TimeDistributed(Dense(BiRNN_UNITS // 2)))
    # crf = CRF(n_tags, name="crf_layer")
    # model.add(crf)

    model.compile('adam',
                  run_eagerly=True,   # 使用动态图
                  loss={'crf_layer': crf.crf_loss},
                  metrics=[crf.crf_accuracy]
                  )
    # loss={'crf_layer': crf_loss},
    # )
    # model.build()
    return model
