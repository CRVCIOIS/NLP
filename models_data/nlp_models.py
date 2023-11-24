from gensim import models, corpora
import gensim.downloader as api
import os

class NLP_Models:
    def __init__(self, vectorized_text, dictionary):
        self.vectorized_text = vectorized_text
        self.dictionary = dictionary
    
    def train_tfidf(self, no_save=False) -> models.TfidfModel:
        tfidf = models.TfidfModel(self.vectorized_text, normalize=True)
        if no_save == False:
            self.save(tfidf, "tfidf")
        return tfidf
            
    def train_lsi(self, topics=10):
        lsi = models.LsiModel(self.vectorized_text, id2word=self.dictionary, num_topics=topics)
        self.save(lsi, "lsi")
        return lsi

    def train_lda(self, topics=2):
        lda = models.LdaModel(self.vectorized_text, id2word=self.dictionary, num_topics=topics)
        self.save(lda, "lda")
        return lda

    def train_okapi(self):
        okapi = models.OkapiBM25Model(self.vectorized_text)
        self.save(okapi, "okapi")
        return okapi
    
    def train_Rp(self, topics=2):
        # Not sure if this is correct, may need to think more about tfidf?
        tfidf_corpus = self.train_tfidf(no_save=True)[self.vectorized_text]

        rp = models.RpModel(tfidf_corpus, num_topics=topics)

        self.save(rp, "rp")
        return rp
    
    def train_Hdp(self):
        hdp = models.HdpModel(self.vectorized_text, id2word=self.dictionary)

        self.save(hdp, "hdp")
        return hdp
    
    def save(self, model, name: str):
        if not os.path.exists("saved"):
            os.makedirs("saved")
        model.save(os.path.join("saved", name + ".model"))

    def load(self, model, name: str):
        if not os.path.exists("saved"):
            os.makedirs("saved")
        return model.load(os.path.join("saved", name + ".model"))
    
    def load_pre_trained(self, name: str):
        try:
            return models.KeyedVectors.load_word2vec_format("saved/" + name + "/" + name + ".gz")
        except FileNotFoundError as e:
            print(e, "Model not found on disk")
    
    def download_models_or_datasets(self, name: str) -> str:
        if not os.path.exists("saved"):
            os.makedirs("saved")
        elif os.path.exists("saved/" + name):
            return "saved/" + name + "/" + name + ".gz"
        api.BASE_DIR = "saved"
        return api.load(name, return_path=True)

        
        


