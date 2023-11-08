import gensim
import gensim.models.doc2vec
from gensim_test import train_model, load_model, plot, doc2vec


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# word2vec
if False:
    # model: gensim.models.Word2Vec = train_model()
    # model: gensim.models.Word2Vec = load_model()
    # model: gensim.models.Word2Vec
    # model.estimate_memory()
    # {'vocab': 15067500, 'vectors': 12054000, 'syn1neg': 12054000, 'total': 39175500}
    # plot("plot.png", model)

    wv = model.wv

    wv.similarity('aktie', 'geld')

    wv.similarity('aktie', 'börse')

    wv.similarity('tech', 'talk')

    print(wv.most_similar(positive=['twitter', "elon"], topn=5))

    print(wv.most_similar(positive=['verkaufen'], topn=5))

    print(wv.most_similar(positive=["twitter"], negative=["musk"], topn=20))

    print(wv.most_similar(positive=["twitter"], topn=10))


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# doc2vec

train_file = "/home/robert/projects/rust/word-embeddings-playground/data/doppelgaenger/raw/doc2vec_train.txt"
test_file = "/home/robert/projects/rust/word-embeddings-playground/data/doppelgaenger/raw/doc2vec_test.txt"
train_corpus = list(doc2vec.read_corpus(train_file))
test_corpus = list(doc2vec.read_corpus(test_file, tokens_only=True))


doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
doc2vec_model.build_vocab(train_corpus)
doc2vec_model.train(train_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

fname = "/home/robert/projects/rust/word-embeddings-playground/data/doppelgaenger/doc2vec_model"
doc2vec_model.save(fname)
doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(fname)

vector = doc2vec_model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print(vector)


doc2vec_model.wv.most_similar(positive=["outperform", "earnings"], topn=10)
