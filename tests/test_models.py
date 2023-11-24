import unittest
import sys
import os
sys.path.append("../NLP")
from models_data.nlp_models import NLP_Models
from gensim import models, corpora

class Test_NLP_Models(unittest.TestCase):
    def setUp(self):
        self.vectorized_text = [[(0,1),(1,1)]]
        self.dictionary = corpora.Dictionary([["test", "test2"]])
        self.models = NLP_Models(self.vectorized_text, self.dictionary)

    def test_train_tfidf(self):
        tfidf_model = self.models.train_tfidf()
        self.assertIsInstance(tfidf_model, models.TfidfModel)
        # Add more assertions to validate the behavior of the method
        os.remove("saved/tfidf.model")

    def test_train_lsi(self):
        lsi_model = self.models.train_lsi()
        self.assertIsInstance(lsi_model, models.LsiModel)
        # Add more assertions to validate the behavior of the method
        os.remove("saved/lsi.model")
        os.remove("saved/lsi.model.projection")

    def test_train_lda(self):
        lda_model = self.models.train_lda()
        self.assertIsInstance(lda_model, models.LdaModel)
        # Add more assertions to validate the behavior of the method
        os.remove("saved/lda.model")
        os.remove("saved/lda.model.expElogbeta.npy")
        os.remove("saved/lda.model.id2word")
        os.remove("saved/lda.model.state")


    def test_train_okapi(self):
        okapi_model = self.models.train_okapi()
        self.assertIsInstance(okapi_model, models.OkapiBM25Model)
        # Add more assertions to validate the behavior of the method
        os.remove("saved/okapi.model")

    def test_train_Rp(self):
        rp_model = self.models.train_Rp()
        self.assertIsInstance(rp_model, models.RpModel)
        # Add more assertions to validate the behavior of the method
        os.remove("saved/rp.model")

    def test_train_Hdp(self):
        hdp_model = self.models.train_Hdp()
        self.assertIsInstance(hdp_model, models.HdpModel)
        # Add more assertions to validate the behavior of the method
        os.remove("saved/hdp.model")
    
    def test_save_load(self):
        modelPre = self.models.train_tfidf()
        self.models.save(modelPre, "tfidf")
        modelpost = self.models.load(models.TfidfModel, "tfidf")
        self.assertIsInstance(modelpost, models.TfidfModel)
        os.remove("saved/tfidf.model")

    def test_download_models_or_datasets(self):
        self.models.download_models_or_datasets("glove-twitter-25")
        self.assertTrue(os.path.exists("saved/glove-twitter-25/glove-twitter-25.gz"))
        os.remove("saved/information.json")
        os.remove("saved/glove-twitter-25/glove-twitter-25.gz")
        os.remove("saved/glove-twitter-25/__init__.py")
        os.rmdir("saved/glove-twitter-25")
    
    def test_load_pre_trained(self):
        self.models.download_models_or_datasets("glove-twitter-25")
        model = self.models.load_pre_trained("glove-twitter-25")
        self.assertIsNotNone(model)
        os.remove("saved/information.json")
        os.remove("saved/glove-twitter-25/glove-twitter-25.gz")
        os.remove("saved/glove-twitter-25/__init__.py")
        os.rmdir("saved/glove-twitter-25")


if __name__ == '__main__':
    unittest.main()