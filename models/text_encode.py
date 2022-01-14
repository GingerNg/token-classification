from collections import Counter
import pickle

class SelfVocabEncoder(object):

    def __init__(self, vocab=None, vocab_path=None):
        self.vocab = vocab
        self.vocab_pth = vocab_path
        if self.vocab_pth is not None:
            with open(self.vocab_pth, 'rb') as f:
                self.vocab = pickle.load(f)
        if self.vocab is not None:
            self.vocab_size = len(self.vocab)
        self.word2idx = None

    def construct_vocab(self, parsed_data):
        if self.vocab is None:
            word_counts = Counter(row[0].lower() for sample in parsed_data for row in sample)
            self.vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
            self.vocab = ['pad', 'unk'] + self.vocab
                    # save initial config data
            if self.vocab_pth is not None:
                with open(self.vocab_pth, 'wb') as outp:
                    pickle.dump(self.vocab, outp)
            self.vocab_size = len(self.vocab)


    def encode_ids(self, s):
        if self.word2idx is None:
            self.word2idx = dict((w, i) for i, w in enumerate(self.vocab))
        return [self.word2idx.get(w[0].lower(), self.word2idx["unk"]) for w in s]