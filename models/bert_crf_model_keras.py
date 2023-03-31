import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow_addons.layers import CRF
from models.base import NerModel
from keras.preprocessing.sequence import pad_sequences

# max_len = 384
configuration = BertConfig()

# Save the slow pretrained tokenizer

model_name = "bert-base-uncased"
save_path = os.getenv("USERPTH")+"/data/huggingface/" + model_name
vocab_path = save_path + "/vocab.txt"

# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# slow_tokenizer.save_pretrained(save_path)

slow_tokenizer = BertTokenizer.from_pretrained(save_path)

class BertLayer(layers.Layer):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.encoder = TFBertModel.from_pretrained(save_path)

    def call(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        embedding = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            training=False
        )[0]
        return embedding


def BertCRFModel(EMBED_DIM, vocab, n_tags, maxlen=50, use_crf=True):

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

def create_inputs(cs, targets, maxlen, chunk_tags):
    # print(s)
    cs = cs[0:maxlen]
    targets = targets[0:maxlen]

    # print(cs)
    tokenized_text = ["CLS"] + cs + ['SEP']
    # print(tokenized_text, len(tokenized_text), len(cs))

    input_ids = [slow_tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    # print(input_ids, len(input_ids))
    targets = [chunk_tags.index("O")] + targets + [chunk_tags.index("O")]

    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    padding_length = maxlen + 2 - len(input_ids)
    if padding_length > 0:  # pad
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([chunk_tags.index("O")] * padding_length)
    return input_ids, token_type_ids, attention_mask, targets

class Ner(NerModel):
    def __init__(self, chunk_tags, vocab_size=1000, emb_dim=100, model_path=None, emb_use_pretrained=0) -> None:
        super().__init__()
        use_crf = True
        n_tags = len(chunk_tags)
        self.chunk_tags = chunk_tags
        self.model = BertCRFModel(emb_dim, vocab_size, n_tags, use_crf=use_crf)
        self.model.summary()
        self.model_path = model_path
        self.maxlen = 36
        self.encoder = None
        self.lang='zh'


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
            maxlen = max(len(s) for s in data)
        # word2idx = dict((w, i) for i, w in enumerate(vocab))

        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        y_chunk = []
        for ws in data:
            # s = '[CLS]' + "".join([w[0] for w in ws]) + '[SEP]'
            cs = [w[0] for w in ws]
            targets = [self.chunk_tags.index(w[1]) for w in ws]
            input_ids, token_type_ids, attention_mask, targets = create_inputs(cs, targets, maxlen, self.chunk_tags)
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
            # print(len(input_ids), len(targets))
            y_chunk.append(targets)

        y_chunk = np.asarray(y_chunk)
        # print(y_chunk)
        # y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
        # y_chunk = pad_sequences(y_chunk, maxlen, value=chunk_tags.index("O"), padding='post')

        if onehot:
            y_chunk = np.eye(len(self.chunk_tags), dtype='float32')[y_chunk]
        else:
            y_chunk = np.expand_dims(y_chunk, 2)
        x = [
            np.asarray(input_ids_list).astype(np.float32),
            np.asarray(token_type_ids_list).astype(np.float32),
            np.asarray(attention_mask_list).astype(np.float32),
        ]
        return x, y_chunk


    def load_model(self):
        pass

    def predict(self):
        pass

    def batch_predict(self):
        pass

    def evaluate(self):
        pass