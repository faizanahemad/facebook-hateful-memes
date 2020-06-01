import spacy
import numpy as np
import pytextrank
from collections import defaultdict
import torch
from facebook_hateful_memes_detector.utils import stack_and_pad_tensors


from joblib import Parallel, delayed


def get_pytextrank_wc_keylen(spacy_doc):
    tokens = [str(x) for x in spacy_doc]
    phrases = [(p.text.split(), p.count) for p in spacy_doc._.phrases]
    pd = defaultdict(dict)
    # tokens = spacy_doc["tokens"]
    # phrases = spacy_doc["phrases"]
    for t, count in phrases:
        c = np.clip(count, 1, 7)
        pd[len(t)][tuple(t)] = c


    from more_itertools import windowed
    key_occ_cnt = [0] * len(tokens)
    key_wc = [0] * len(tokens)

    for w in range(1, 4):
        for pos, words in enumerate(windowed(tokens, w)):
            if words in pd[w]:
                c = pd[w][words]
                for i in range(pos, pos + w):
                    key_occ_cnt[i] = c
                    key_wc[i] = w

    return torch.tensor(key_wc), torch.tensor(key_occ_cnt)


def get_rake_nltk_phrases(rake, text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:10]

def get_rake_nltk_wc(tokens, phrases):
    phrases = [tuple(p.split()) for p in phrases]
    pd = defaultdict(set)
    for t in phrases:
        pd[len(t)].add(t)


    from more_itertools import windowed
    key_wc = [0] * len(tokens)

    for w in range(1, 4):
        for pos, words in enumerate(windowed(tokens, w)):
            if words in pd[w]:
                for i in range(pos, pos + w):
                    key_wc[i] = w

    return torch.tensor(key_wc)

def get_yake_embedding(text, kwe):
    keywords = kwe.extract_keywords(text)



if __name__=="__main__":
    import time
    text = 'have you ever studied the history of the jews? did you know that they have always banded together as a tribe, infiltrated governments, monopolized the financial systems of nations instigated wars and intentionally created chaos in societies? the jews have mass murdered millions of non- jews over the centuries they have seized control of the media so you will never find out study the history of the jews!'
    text = 'have you ever studied the history of the jews? did you know that they have always banded together as a tribe, infiltrated governments, monopolized the financial systems of nations instigated wars and intentionally created chaos in societies?'
    nlp = spacy.load("en_core_web_lg")
    tr = pytextrank.TextRank(token_lookback=7)
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    doc = nlp(text)
    doc_array = [doc] * 2
    # doc_array = [[(str(p.text), int(p.count)) for p in doc._.phrases] for doc in doc_array]
    # print(type(doc_array[0]),type(doc_array[0][0]), type(doc_array[0][0][0]), type(doc_array[0][0][1]))
    # dc = []
    # for d in doc_array:
    #     phrases = []
    #     tokens = [str(x) for x in d]
    #     for p in d._.phrases:
    #         phrases.append((str(p.text).split(), int(p.count)))
    #     dc.append(dict(phrases=phrases, tokens=tokens))
    times = 10
    s = time.time()
    for _ in range(times):
        results = [get_pytextrank_wc_keylen(i) for i in doc_array]
    e = (time.time() - s) / times
    print("Mapper No threads Time Taken =", e)

    # s = time.time()
    # for _ in range(times):
    #     results = Parallel(n_jobs=1)(delayed(get_wc_keylen)(i) for i in dc)
    # e = (time.time() - s) / times
    # print("Joblib 1 thread Time Taken =", e)
    #
    # s = time.time()
    # for _ in range(times):
    #     results = Parallel(n_jobs=4)(delayed(get_wc_keylen)(i) for i in dc)
    # e = (time.time() - s) / times
    # print("Joblib 4 thread Time Taken =", e)

    key_wc, key_occ_cnt = zip(*results)
    print(len(key_wc), len(key_occ_cnt))
    print(key_wc, "\n", key_occ_cnt)

    print(stack_and_pad_tensors(key_wc, 64))
    print(stack_and_pad_tensors(key_occ_cnt, 64))
    print(stack_and_pad_tensors(key_wc, 64).shape)
    print(stack_and_pad_tensors(key_occ_cnt, 64).shape)

