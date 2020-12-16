import pickle

import numpy as np

from load_data import read_files, word_tokenize, init_matrix


def convert(label):
    return 1 if label == "true" else 0


def get_data_word2vec(article_size=50, sentence_size=20):
    # set path for data
    data_path = 'data/'

    text_file = data_path + 'articles-training-byarticle.xml'
    label_file = data_path + "ground-truth-training-byarticle.xml"

    # read in data and glove vectors
    content_dic = read_files(text_file, label_file)
    word_dic = pickle.load(open("utils/word_dict.pkl", "rb"))

    y_data = np.array([convert(label) for label in content_dic["label"]], dtype=np.long)

    # load dataset
    articles = np.zeros((len(content_dic["article"]), article_size, sentence_size))
    for article_id, article in enumerate(content_dic["article"]):
        article = [word_tokenize(sentence) for sentence in article]
        articles[article_id] = init_matrix(word_dic, article, (article_size, sentence_size))
    titles = np.array([init_matrix(word_dic, word_tokenize(t), (1, sentence_size)).squeeze()
                       for t in content_dic["title"]])
    articles = articles.reshape((-1, article_size * sentence_size))

    x_data = np.concatenate((titles, articles), axis=1)
    return x_data, y_data

accuracy = pickle.load(open("accuracy.pkl", "rb"))
