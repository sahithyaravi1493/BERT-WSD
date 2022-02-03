from turtle import pos
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from demo_model import get_predictions, load_model
import spacy
import re

# from pywsd.lesk import simple_lesk
model_dir = "/ubc/cs/research/nlp/sahiravi/BERT-WSD/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"
nlp = spacy.load("en_core_web_md")
dir = '/ubc/cs/research/nlp/sahiravi/datasets/caches'
# nltk.download('averaged_perceptron_tagger', download_dir=dir)
nltk.download('omw-1.4', download_dir=dir)
nltk.download('wordnet', download_dir=dir)
nltk.download('stopwords', download_dir=dir)
nltk.data.path.append(dir)
stopwords = nltk.corpus.stopwords.words('english')
# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "agent", "expl","csubj"}
POS_ALLOWED = {"NOUN"} #{"VERB", "NOUN", "ADJ"}

# def pos_to_wordnet_pos(penntag, returnNone=False):
#     morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
#                     'VB':wn.VERB, 'RB':wn.ADV}
#     try:
#         return morphy_tag[penntag[:2]]
#     except:
#         return None if returnNone else ''
sense_key_regex = r"(.*)\%(.*):(.*):(.*):(.*):(.*)"
synset_types = {1:'n', 2:'v', 3:'a', 4:'r', 5:'s'}

def synset_from_sense_key(sense_key):
    lemma, ss_type, lex_num, lex_id, head_word, head_id = re.match(sense_key_regex, sense_key).groups()
    ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
    return wn.synset(ss_idx)


def get_synonyms(word, tag=wn.NOUN):
    for synset in wn.synsets(word, pos=tag):
        for lemma in synset.lemmas():
            yield lemma.name()

def get_hypernyms(sense, tag=wn.NOUN):
    for synset in sense.hypernyms():
        for lemma in synset.lemmas():
            yield lemma.name()

def disambiguate(sentence, word, method = "frequency"):
    sense = None
    if method == "lesk":
        sense = simple_lesk(sentence, word, "n")
    elif method == "frequency":
        if wn.synsets(word):
            sense = wn.synsets(word)[0]
    return sense


def isvalid(t):
    return (len(t.text) > 2) and (t.lemma_ not in stopwords) and (t.pos_ in POS_ALLOWED)


def load_text(path):
    with open(path) as f:
        input = f.readlines()
    return input

def get_bert_predictions(sentence, word):
    model, tokenizer = load_model(model_dir)
    word_tgt = f"[TGT]{word}[TGT]"
    p = get_predictions(model, tokenizer, sentence.replace(word,word_tgt))
    return p

o = get_bert_predictions("The sheep are on the farm", "farm")
best = o[0][0]
print(synset_from_sense_key(best))