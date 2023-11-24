import os
from gensim.models import KeyedVectors

HERE = os.path.dirname(os.path.abspath(__file__))
GLOVE_FILES_PATH = HERE + "/glove_files"

if __name__ == "__main__":
    models = []
    for fpath in os.listdir(GLOVE_FILES_PATH):
        if not os.path.isfile(fpath):
            continue
        models.append(KeyedVectors.load_word2vec_format(fpath, binary=False, no_header=True))
