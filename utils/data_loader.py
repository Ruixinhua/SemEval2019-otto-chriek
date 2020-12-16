import pickle
from nltk.tokenize import sent_tokenize
from utils.utils import *
from lxml.etree import iterparse


def read_files(text_file, label_file):
    content = {"article": [], "title": [], "label": []}
    with open(label_file) as labelFile:
        xml.sax.parse(labelFile, GroundTruthHandler(content["label"]))

    for event, elem in iterparse(text_file):
        if elem.tag == "article":
            title = elem.attrib['title']
            text = "".join(elem.itertext())
            title = cleanQuotations(title)
            text = cleanQuotations(text)
            text = cleanText(fixup(text))
            text = ' '.join(text.split())
            content["title"].append(title)
            content["article"].append(sent_tokenize(text))
            elem.clear()

    return content


def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def init_matrix(word_dict, data, shape):
    matrix = np.zeros(shape, dtype="int32")
    for index in range(min(matrix.shape[0], len(data))):
        content = data[index]
        for word_index in range(min(len(content), matrix.shape[1])):
            if content[word_index] in word_dict:
                matrix[index, word_index] = word_dict[content[word_index].lower()]
    return matrix


def convert(label):
    return 1 if label == "true" else 0


def get_data_word2vec(article_size=50, sentence_size=20, data_dir="data/", word_dic_dir="utils/"):
    # set path for data
    text_file = data_dir + 'articles-training-byarticle.xml'
    label_file = data_dir + "ground-truth-training-byarticle.xml"

    # read in data and glove vectors
    content_dic = read_files(text_file, label_file)
    word_dic = pickle.load(open(f"{word_dic_dir}word_dict.pkl", "rb"))

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

