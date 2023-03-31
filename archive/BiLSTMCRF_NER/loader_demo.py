import os

from loader import load_sentences

if __name__ == '__main__':
    train_sentences = load_sentences(os.path.join("data", "example.train"), True, False)
    print(train_sentences)