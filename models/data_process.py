import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import pickle
import random
class Processor(object):
    def __init__(self):
        pass

    def process_data(self, s):
        return s


class ConllProcessor(Processor):
    def __init__(self):
        super().__init__()

    def _process_data(self, data, vocab, chunk_tags, maxlen=50, onehot=False):
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


    def process_data(self, parsed_data):
        """[load data in conll format]

        Args:
            train_pth (str, optional): [description]. Defaults to
            test_pth (str, optional): [description]. Defaults to

        Returns:
            [type]: [description]
        """
        # parsed_data = self._parse_data(open(data_pth, 'r'))
        word_counts = Counter(row[0].lower() for sample in parsed_data for row in sample)
        vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
        vocab = ['pad', 'unk'] + vocab
        # chunk_tags = ['O', 'B-BRD', 'I-BRD']
        with open('./data/CCKS2021中文NLP地址要素解析/tags.pkl', 'rb') as f:
            chunk_tags = pickle.load(f)

        # save initial config data
        with open('saved_model/ccks2021_config.pkl', 'wb') as outp:
            pickle.dump((vocab, chunk_tags), outp)

        print("num of chunk_tags: {}".format(len(chunk_tags)))
        processd_data = self._process_data(parsed_data, vocab, chunk_tags)
        return processd_data, (vocab, chunk_tags)