from nltk.tokenize import sent_tokenize
from utils.utils import *
from lxml.etree import iterparse
import pandas as pd


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


def read_glove(path, dim):
    '''
    read the glove vectors from path with dimension dim
    '''
    df = pd.read_csv(path + 'glove.6B.' + str(dim) + 'd.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}
    return glove


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


