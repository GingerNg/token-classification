import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow_addons.layers import CRF

# max_len = 384
configuration = BertConfig()

# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
save_path = "bert_base_chinese/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)


class BertLayer(layers.Layer):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.encoder = TFBertModel.from_pretrained("bert-base-chinese")

    def call(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        embedding = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            training=False
        )[0]
        return embedding


def create_model(EMBED_DIM, vocab, n_tags, maxlen=50, use_crf=True):

    input_ids = layers.Input(shape=(maxlen+2,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(maxlen+2,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(maxlen+2,), dtype=tf.int32)

    bert_layer = BertLayer()
    bert_layer.trainable = False
    embedding = bert_layer([input_ids, token_type_ids, attention_mask])

    dropout1 = layers.Dropout(0.1)
    out = layers.Dense(256, activation="relu")(dropout1(embedding))

    if use_crf:
        crf = CRF(n_tags, name="output_1")
        out = crf(out)
        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=out,
        )
        model.compile('adam',
                      loss={'output_1': crf.crf_loss},
                      metrics=[crf.crf_accuracy]
                      )
    else:
        dropout2 = layers.Dropout(0.1)
        out = layers.Dense(n_tags, activation="softmax")(dropout2(out))

        model = keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=out,
        )
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy", metrics=["accuracy"])
    return model
