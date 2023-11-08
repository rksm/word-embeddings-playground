import os
from gensim.test.utils import datapath
from gensim import utils
import gensim

dir = "/home/robert/projects/rust/word-embeddings-playground/data/doppelgaenger/"
corpus_txt = dir + "raw/full.txt"

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath(corpus_txt)
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


def train_model() -> gensim.models.Word2Vec:
    corpus = MyCorpus()

    # parameters:
    #   min_count: ignore all words with total frequency lower than this
    #   vector_size: dimensionality of the word vectors
    #   workers: use this many worker threads
    model = gensim.models.Word2Vec(sentences=corpus, min_count=3, vector_size=100, workers=12, compute_loss=True, epochs=10)
    print(f"loss: {model.get_latest_training_loss()}")
    model.save(dir + "full.model")

    return model

def load_model() -> gensim.models.Word2Vec:
    return gensim.models.Word2Vec.load(dir + "full.model")


def eval(model: gensim.models.Word2Vec):
    score, sections = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

    print(f"score: {score}")
    for section in sections:
        correct = len(section['correct'])
        incorrect = len(section['incorrect'])
        all = correct + incorrect
        print(f"{section['section']}: {correct}/{all}")
