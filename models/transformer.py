import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, BatchNormalization, Dropout
from CRF import CRF
# from datasets import load_dataset
# from collections import Counter
# from conlleval import evaluate


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(
            input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen

    def call(self, inputs):
        # maxlen = tf.shape(inputs)[-1]
        maxlen = self.maxlen
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_model(EMBED_DIM, vocab, n_tags, maxlen=50, use_crf=True):
    embed_dim = EMBED_DIM
    num_heads = 2
    ff_dim = 32
    vocab_size = len(vocab)
    model = Sequential()
    model.add(TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim))
    # model.add(Embedding(len(vocab), EMBED_DIM,
    #                     embeddings_initializer=keras.initializers.Orthogonal(gain=1.0, seed=1),
    #                     trainable=True)
    #           )  # Random embedding
    model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(ff_dim, activation="relu"))
    model.add(layers.Dropout(0.1))
    # model.add(TimeDistributed(Dense(BiRNN_UNITS // 2)))
    if use_crf:
        crf = CRF(n_tags, name="output_1")
        model.add(crf)
        model.compile('adam', loss={'output_1': crf.crf_loss},
                    metrics=[crf.crf_accuracy,
                            #   tfa.metrics.F1Score(average='micro', num_classes=59)
                            #   tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='micro')
                            ]
                    )
    else:
        model.add(layers.Dense(n_tags, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    input_shape = (None, 50)
    model.build(input_shape)
    return model
