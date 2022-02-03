
from helpers import *
import pandas as pd
from tqdm import tqdm

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

