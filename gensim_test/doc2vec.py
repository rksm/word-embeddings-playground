import gensim
import smart_open

def read_corpus(fname: str, tokens_only=False):
    with smart_open.open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def run():
    train_file = "/home/robert/projects/rust/word-embeddings-playground/data/doppelgaenger/raw/doc2vec_train.txt"
    test_file = "/home/robert/projects/rust/word-embeddings-playground/data/doppelgaenger/raw/doc2vec_test.txt"
    train_corpus = list(read_corpus(train_file))
    test_corpus = list(read_corpus(test_file, tokens_only=True))
