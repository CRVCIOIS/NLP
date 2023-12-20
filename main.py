import spacy
from spacy import displacy
from preprocess import parse_HTML

def named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    displacy.serve(doc, style="ent")

def display(text,style):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    displacy.serve(doc, style=style)

if __name__ == "__main__":
    PATH = "scraped\\file.htm"
    text = parse_HTML(PATH).get_text()
    display(text, "dep")