import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import pickle
import random

maxlen = 36


def load_data(train_pth='data/CCKS2021中文NLP地址要素解析/train_human.conll', test_pth='data/CCKS2021中文NLP地址要素解析/dev.conll'):
    """[load data in conll format]

    Args:
        train_pth (str, optional): [description]. Defaults to
        test_pth (str, optional): [description]. Defaults to

    Returns:
        [type]: [description]
    """
    train = _parse_data(open(train_pth, 'r'))
    # train = _parse_data(open('data/CCKS2021中文NLP地址要素解析/train.conll', 'r'))
    test = _parse_data(open(test_pth, 'r'))
    # print(train)
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    vocab = ['pad', 'unk'] + vocab
    # chunk_tags = ['O', 'B-BRD', 'I-BRD']
    with open('./data/CCKS2021中文NLP地址要素解析/tags.pkl', 'rb') as f:
        chunk_tags = pickle.load(f)

    # save initial config data
    with open('saved_model/ccks2021_config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    print("num of chunk_tags: {}".format(len(chunk_tags)))

    random.shuffle(train)
    cutoff = int(len(train) * 0.8)
    val = train[cutoff:]
    train = train[0:cutoff]

    train = _process_data(train, vocab, chunk_tags)
    val = _process_data(val, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
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


def _process_data(data, vocab, chunk_tags, maxlen=50, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))

    x = [[word2idx.get(w[0].lower(), word2idx["unk"]) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    # x = pad_sequences(x, maxlen, value=word2idx["pad"])
    # y_chunk = pad_sequences(y_chunk, maxlen, value=chunk_tags.index("O"))

    x = pad_sequences(x, maxlen, value=word2idx["pad"], padding='post')
    y_chunk = pad_sequences(y_chunk, maxlen, value=chunk_tags.index("O"), padding='post')
    print("y_chunk shape:", y_chunk.shape)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        pass
        # y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=50):
    if maxlen is None:
        maxlen = len(data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    # x = pad_sequences([x], maxlen, value=word2idx["pad"])
    x = pad_sequences([x], maxlen, value=word2idx["pad"], padding='post')
    return x, length
