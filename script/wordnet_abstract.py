
import pandas as pd
from tqdm import tqdm
# from turtle import pos
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

import spacy
import re

# from pywsd.lesk import simple_lesk
BERT_WSD = True
model_dir = "/ubc/cs/research/nlp/sahiravi/BERT-WSD/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"
from demo_model import get_predictions, load_model
model, tokenizer = load_model(model_dir)


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

# def synset_from_sense_key(sense_key):
#     lemma, ss_type, lex_num, lex_id, head_word, head_id = re.match(sense_key_regex, sense_key).groups()
#     ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
#     return wn.synset(ss_idx)


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
    if BERT_WSD:
        sense = get_bert_predictions(sentence, word)
    elif method == "lesk":
        sense = lesk(sentence, word, "n")
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
    out = None
    word_tgt = f"[TGT]{word}[TGT]"
    p = get_predictions(model, tokenizer, sentence.replace(word,word_tgt))
    if p:
        print(p)
        out = p[0][1]
    else:
        if wn.synsets(word):
            out = wn.synsets(word)[0]
        
    return out


# wup_similarity
# extract the subject, object and verb from the input
def extract_svo(doc):
    sub = []
    at = []
    ve = []
    all = set()
    for token in doc:
        # is this a verb?
        if token.pos_ == "VERB":
            ve.append(token.text)
        # is this the object?
        if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
            at.append(token.text)
            all.add(token.text)
        # is this the subject?
        if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
            sub.append(token.text)
            all.add(token.text)
    #return " ".join(sub).strip().lower(), " ".join(ve).strip().lower(), " ".join(at).strip().lower()
    return sub, ve, at, all

def extract_pos_based(doc):
    out = []
    for token in doc:
        if isvalid(token):
            out.append(token.text)
    return out

def construct_abstractions(sentence, extract_method="pos", abstract_method="hypernyms"):
    doc = nlp(sentence)
    if extract_method == "svo":
        subject, verb, attribute, all_words = extract_svo(doc)
    elif extract_method == "pos":
        all_words = extract_pos_based(doc)


    abstraction_map = {}
    abs_sentences = []
    # print(all_words)
    for word in all_words:
        if abstract_method == "synonyms":
            unique = set(synonym for synonym in get_synonyms(word) if synonym != word)
            abstraction_map[word] = list(unique)[:5]
        elif abstract_method == "hypernyms":
            sense = disambiguate(sentence, word)
            if sense is not None:
                unique = sorted(set(h for h in get_hypernyms(sense) if h != word))
                abstraction_map[word] = unique
            
    for word in abstraction_map:
        for syn in abstraction_map[word]:
            out = sentence
            abs_sentences.append(out.replace(word, syn))

    return abs_sentences
        

def all_sentence_abstractions(text):
    abstracted_sentences = []
    indices = []
    for i in tqdm(range(len(text))):
        sentences = construct_abstractions(text[i], extract_method="pos", abstract_method="synonyms")
        abstracted_sentences.extend(sentences)
        indices.extend([i]*len(sentences))
    df = pd.DataFrame()
    df["gen_id"] = indices
    df["abstractions"] = abstracted_sentences
    return df

# gather the user input and gather the info
if __name__ == "__main__":
    print(" Generate Abstractions for a sample input based on synonyms from wordnet")
    sent = "A cat and its furry companions on a couch."
    abstractions = construct_abstractions(sent, extract_method="pos", abstract_method="hypernyms")
    print(abstractions)



    # Now process generated text from T5 into a dataframe

    # split = 'valid'
    # root = '/ubc/cs/research/nlp/sahiravi/generative_csr/datasets/commongen/'
    # source_path = f'{root}/{split}.source'
    # target_path = f'{root}/{split}.target'
    # output_path = f'/ubc/cs/research/nlp/sahiravi/generative_csr/outputs/commongen_K50_P0.8/{split}.txt'
    # input = load_text(source_path)
    # output = load_text(output_path)
    # target = load_text(target_path)
    # n_sentences = 5
    # concepts = []
    # targets = []
    # cids = []

    # for i in range(len(input)):
    #     concepts.extend([input[i]]*n_sentences)
    #     cids.extend([i]*n_sentences)
    # for line in target:
    #     targets.extend([line]*n_sentences)

    # print(len(output), len(targets), len(concepts), len(cids))
    # gen_text = pd.DataFrame()
    # gen_text["generated"] = output
    # gen_text["concepts"] = concepts
    # gen_text["targets"] = targets
    # gen_text["concept_set_idx"] = cids
    # gen_text["gen_id"] = list(range(len(output)))
  
    # # generate and save abstractions
    # df_abstract = all_sentence_abstractions(gen_text["generated"].values)
    # df_abstract.to_csv("abstracted_df.csv")

