import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import pickle
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import numpy as np
import random


encoder = BertWordPieceTokenizer("bert_base_chinese/vocab.txt", lowercase=True)
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def load_data():
    train = _parse_data(open('data/CCKS2021中文NLP地址要素解析/train_human.conll', 'r'))
    # train = _parse_data(open('data/CCKS2021中文NLP地址要素解析/train.conll', 'r'))
    test = _parse_data(open('data/CCKS2021中文NLP地址要素解析/dev.conll', 'r'))

    random.shuffle(train)
    cutoff = int(len(train) * 0.8)
    val = train[cutoff:]
    train = train[0:cutoff]
    # print(train)

    # chunk_tags = ['O', 'B-BRD', 'I-BRD']
    with open('./data/CCKS2021中文NLP地址要素解析/tags.pkl', 'rb') as f:
        chunk_tags = pickle.load(f)

    vocab = []
    # save initial config data
    with open('saved_model/ccks2021_config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, chunk_tags)
    test = _process_data(test, chunk_tags)
    val = _process_data(val, chunk_tags)
    return train, val, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read()
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    for s in data:
        for w in s:
            if len(w) < 2:
                print(w)
    fh.close()
    return data


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


def _process_data(data, chunk_tags, maxlen=50, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    # word2idx = dict((w, i) for i, w in enumerate(vocab))

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    y_chunk = []
    for ws in data:
        # s = '[CLS]' + "".join([w[0] for w in ws]) + '[SEP]'
        cs = [w[0] for w in ws]
        targets = [chunk_tags.index(w[1]) for w in ws]
        input_ids, token_type_ids, attention_mask, targets = create_inputs(cs, targets, maxlen, chunk_tags)
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
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    x = [
        np.asarray(input_ids_list).astype(np.float32),
        np.asarray(token_type_ids_list).astype(np.float32),
        np.asarray(attention_mask_list).astype(np.float32),
    ]
    return x, y_chunk


def process_data(data, vocab, maxlen=50):
    if maxlen is None:
        maxlen = len(data)
    length = len(data)
    cs = list(data)
    cs = cs[0:maxlen]
    tokenized_text = ["CLS"] + cs + ['SEP']
    # print(tokenized_text, len(tokenized_text), len(cs))
    input_ids = [slow_tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    padding_length = maxlen+2 - len(input_ids)
    if padding_length > 0:  # pad
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    x = [
        np.asarray([input_ids]).astype(np.float32),
        np.asarray([token_type_ids]).astype(np.float32),
        np.asarray([attention_mask]).astype(np.float32),
    ]
    return x, length
