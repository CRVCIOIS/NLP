from models_data.nlp_models import NLP_Models
from input.preprocess import InputPreprocess
from gensim import corpora, models, similarities
from input.nace_identifier import parse_codes
from collections import defaultdict
import gzip


def train():
    preprocess = InputPreprocess()
    nace_codes = parse_codes("codes.json")
    (tokenized_corpus, dictionary) = preprocess.execute(nace_codes.keys())
    vectorized_corpus = preprocess.doc2bow(tokenized_corpus, dictionary)
    
    preprocess.write_tokenized_corpus_as_vectorized_corpus_to_file(tokenized_corpus, "nace_tokenized_corpus")
    dictionary.save("nace_dictionary")
    
    nlp_models = NLP_Models(vectorized_corpus, dictionary)
    nlp_models.train_lsi(100)
    nlp_models.train_lda(100)

def test():
    # load pre-trained model
    nlp_models = NLP_Models(None, None)
    lsi = nlp_models.load(models.LsiModel, "lsi")
    lda = nlp_models.load(models.LdaModel, "lda")
    
    # load dictionary
    dictionary = corpora.Dictionary.load("nace_dictionary")
    # for key, value in dictionary.items():
    #     print(key, value)
    
    # load vectorized corpus
    preprocess = InputPreprocess("nace_tokenized_corpus")
    vectorized_corpus = preprocess.read_vectorized_corpus_from_file("nace_tokenized_corpus")
    
    # create index
    lsi_index = similarities.MatrixSimilarity(lsi[vectorized_corpus], num_features=100)
    lda_index = similarities.MatrixSimilarity(lda[vectorized_corpus], num_features=100)
    
    
    # get sims of nace codes
    nace_codes = parse_codes("codes.json")
    nace_codes = list(nace_codes.keys())
    
    # transform query to space of models
    query = ["Manufacture of premium metal-cutting tools, providing high-end services and machining technology."]
    query, _ = preprocess.execute(query)
    print(query)
    query = preprocess.doc2bow(query, dictionary)[0]
    print(query)
    
    lsi_query = lsi[query]
    lda_query = lda[query]
    
    # get similarities of query
    lsi_sims = lsi_index[lsi_query]
    lda_sims = lda_index[lda_query]
    
    
    # print results
    print("LSI:")
    sims = sorted(enumerate(lsi_sims), key=lambda item: -item[1])
    for doc_position, doc_score in sims:
        if (doc_score > 0.5):
             print(doc_score, nace_codes[doc_position])
        
    print("LDA:")
    sims = sorted(enumerate(lda_sims), key=lambda item: -item[1])
    for doc_position, doc_score in sims:
         if (doc_score > 0.5):
             print(doc_score, nace_codes[doc_position])
    
    
    
    
    
    
    

def main():
    train()
    test()


if __name__ == "__main__":
    main()

    
