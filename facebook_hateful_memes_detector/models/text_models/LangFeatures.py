import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BytePairEmbeddings, CharacterEmbeddings, WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
import re
import json
import csv
import numpy as np

from scipy.special import softmax
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

import spacy

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_pos_tag_indices, pad_tensor, \
    get_penn_treebank_pos_tag_indices, get_all_tags, has_words, ExpandContract
from ...utils import get_universal_deps_indices, has_digits
from ..external import get_pytextrank_wc_keylen, get_rake_nltk_wc, get_rake_nltk_phrases
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer
from .Fasttext1DCNN import Fasttext1DCNNModel
import pytextrank
import gensim.downloader as api
from operator import itemgetter
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class LangFeaturesModel(Fasttext1DCNNModel):
    def __init__(self, classifier_dims, num_classes, embedding_dims,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False,
                 **kwargs):
        super(LangFeaturesModel, self).__init__(classifier_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                                internal_dims, n_layers,
                                                featurizer, final_layer_builder,
                                                n_tokens_in, n_tokens_out, use_as_super=True, **kwargs)
        assert "capabilities" in kwargs
        capabilities = kwargs["capabilities"]
        assert "key_phrases" not in capabilities or ("key_phrases" in capabilities and "spacy" in capabilities)
        use_layer_norm = kwargs["use_layer_norm"] if "use_layer_norm" in kwargs else True
        self.capabilities = capabilities
        embedding_dim = 8
        cap_to_dim_map = {"spacy": 128, "snlp": 32,
                          "key_phrases": 64, "nltk": 192, "full_view": 64,
                          "tmoji": 32, "ibm_max": 16, "gensim": 256, "fasttext_crawl": 192}
        all_dims = sum([cap_to_dim_map[c] for c in capabilities])
        self.cap_to_dim_map = cap_to_dim_map
        self.all_dims = all_dims

        if "spacy" in capabilities:
            tr = pytextrank.TextRank(token_lookback=7)
            self.nlp = spacy.load("en_core_web_lg", disable=[])
            self.nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
            spacy_in_dims = (96*2) + (11 * embedding_dim) + 2
            self.spacy_nn = ExpandContract(spacy_in_dims, cap_to_dim_map["spacy"], dropout, use_layer_norm=use_layer_norm, groups=(2, 4))

        if "fasttext_crawl" in capabilities:
            self.bpe = BPEmb(dim=200)
            self.cngram = CharNGram()
            fasttext_crawl_file = kwargs["fasttext_crawl_file"] if "fasttext_crawl_file" in kwargs else "crawl-300d-2M-subword.bin"
            self.crawl = fasttext.load_model(fasttext_crawl_file)
            self.crawl_nn = ExpandContract(200+300+100, cap_to_dim_map["fasttext_crawl"], dropout, use_layer_norm=use_layer_norm, groups=(4, 4))

        if "gensim" in capabilities:
            gensim = [api.load("glove-twitter-50"), api.load("glove-wiki-gigaword-50"),
                      api.load("word2vec-google-news-300"), api.load("conceptnet-numberbatch-17-06-300")]
            self.gensim = gensim
            self.gensim_nn = ExpandContract(700, cap_to_dim_map["gensim"], dropout, use_layer_norm=use_layer_norm, groups=(4, 4))

        if "full_view" in capabilities:
            full_sent_in_dims = 300
            self.full_sent_nn = ExpandContract(full_sent_in_dims, cap_to_dim_map["full_view"], dropout, use_layer_norm=use_layer_norm, groups=(4, 4))

        if "snlp" in capabilities:
            import stanza
            self.snlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,ner', use_gpu=False,
                                    pos_batch_size=2048)
            self.snlp_nn = ExpandContract(embedding_dim * 5, cap_to_dim_map["snlp"], dropout,
                                          use_layer_norm=use_layer_norm)
        if "key_phrases" in capabilities:
            import yake
            from multi_rake import Rake
            self.kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9,
                                                 dedupFunc='seqm', windowsSize=3,
                                                 top=10, features=None)

            self.key_occ_cnt_pytextrank = nn.Embedding(8, embedding_dim)
            nn.init.normal_(self.key_occ_cnt_pytextrank.weight, std=1 / embedding_dim)
            self.key_wc_pytextrank = nn.Embedding(4, embedding_dim)
            nn.init.normal_(self.key_wc_pytextrank.weight, std=1 / embedding_dim)

            yake_dims = kwargs["yake_dims"] if "yake_dims" in kwargs else 32
            self.yake_dims = yake_dims
            self.yake_nn = ExpandContract(300, yake_dims, dropout, use_layer_norm=use_layer_norm, groups=(2, 2))

            rake_dims = kwargs["rake_dims"] if "rake_dims" in kwargs else 32
            self.rake_dims = rake_dims
            self.rake_nn = ExpandContract(300, rake_dims, dropout, use_layer_norm=use_layer_norm, groups=(2, 2))
            self.rake = Rake(language_code="en")

            keyphrases_dim = 2*embedding_dim + rake_dims + yake_dims
            self.keyphrase_nn = ExpandContract(keyphrases_dim, cap_to_dim_map["key_phrases"], dropout, use_layer_norm=use_layer_norm, groups=(4, 4))

        fasttext_file = kwargs["fasttext_file"] if "fasttext_file" in kwargs else "wiki-news-300d-1M-subword.bin"
        assert fasttext_file is not None
        if fasttext_file is not None:
            self.text_model = fasttext.load_model(fasttext_file)

        self.pdict = get_all_tags()
        self.tag_em = nn.Embedding(len(self.pdict)+1, embedding_dim)
        nn.init.normal_(self.tag_em.weight, std=1 / embedding_dim)

        self.sw_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.sw_em.weight, std=1 / embedding_dim)

        self.sent_start_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.sent_start_em.weight, std=1 / embedding_dim)

        self.is_oov_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.is_oov_em.weight, std=1 / embedding_dim)



        self.has_digit_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.has_digit_em.weight, std=1 / embedding_dim)

        self.is_mask_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.is_mask_em.weight, std=1 / embedding_dim)

        self.w_len = nn.Embedding(16, embedding_dim)
        nn.init.normal_(self.w_len.weight, std=1 / embedding_dim)

        self.wc_emb = nn.Embedding(16, embedding_dim)
        nn.init.normal_(self.wc_emb.weight, std=1 / embedding_dim)

        if "nltk" in capabilities:
            import rake_nltk
            from textblob import TextBlob
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer
            self.stop_words = set(stopwords.words('english'))
            self.rake_nltk = rake_nltk.Rake()
            self.key_wc_rake_nltk = nn.Embedding(4, embedding_dim)
            nn.init.normal_(self.key_wc_rake_nltk.weight, std=1 / embedding_dim)
            self.nltk_sid = SentimentIntensityAnalyzer()
            self.vader_sid = VaderSentimentIntensityAnalyzer()
            in_dims = 310 + 5 * embedding_dim
            self.nltk_nn = ExpandContract(in_dims, cap_to_dim_map["nltk"], dropout, use_layer_norm=use_layer_norm, groups=(2, 4))

        if "ibm_max" in capabilities:
            from ..external import ModelWrapper
            self.ibm_max = ModelWrapper()
            for p in self.ibm_max.model.parameters():
                p.requires_grad = False
            self.ibm_nn = ExpandContract(6, cap_to_dim_map["ibm_max"], dropout, use_layer_norm=use_layer_norm, groups=(1, 1))

        if "tmoji" in capabilities:
            from torchmoji.sentence_tokenizer import SentenceTokenizer
            from torchmoji.model_def import torchmoji_emojis
            from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
            with open(VOCAB_PATH, 'r') as f:
                maxlen = self.n_tokens_in
                self.vocabulary = json.load(f)
                self.st = SentenceTokenizer(self.vocabulary, maxlen)
                self.tmoji = torchmoji_emojis(PRETRAINED_PATH)
                for p in self.tmoji.parameters():
                    p.requires_grad = False
            self.tm_nn = ExpandContract(64, cap_to_dim_map["tmoji"], dropout, use_layer_norm=use_layer_norm, groups=(1, 1))

        self.contract_nn = ExpandContract(self.all_dims, embedding_dims, dropout,
                                          use_layer_norm=True, unit_norm=False, groups=(4, 4))
        if not use_as_super:
            if featurizer == "cnn":
                self.featurizer = CNN1DFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims, internal_dims, None, gaussian_noise, dropout)
            elif featurizer == "gru":
                self.featurizer = GRUFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "basic":
                self.featurizer = BasicFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims,
                                                  internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

    def get_one_crawl_sentence_vector(self, tm, sentence):
        tokens = word_tokenize(sentence)
        if isinstance(tm, fasttext.FastText._FastText):
            result = torch.tensor([tm[t] for t in tokens])
        elif isinstance(tm, torchnlp.word_to_vector.char_n_gram.CharNGram):
            result = torch.stack([tm[t] for t in tokens])
        else:
            result = tm[tokens]
        return result

    def get__crawl_word_vectors(self, texts: List[str]):
        bpe = self.bpe
        cngram = self.cngram
        tm = self.crawl
        n_tokens_in = self.n_tokens_in
        result = stack_and_pad_tensors([self.get_one_crawl_sentence_vector(tm, text) for text in texts], n_tokens_in)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        res2 = stack_and_pad_tensors([self.get_one_crawl_sentence_vector(bpe, text) for text in texts], n_tokens_in)
        res2 = res2 / res2.norm(dim=2, keepdim=True).clamp(min=1e-5)
        res3 = stack_and_pad_tensors([self.get_one_crawl_sentence_vector(cngram, text) for text in texts], n_tokens_in)
        res3 = res3 / res3.norm(dim=2, keepdim=True).clamp(min=1e-5)
        result = torch.cat([result, res2, res3], 2)
        result = self.crawl_nn(result)
        return result

    def get_torchmoji_probas(self,  texts: List[str]):
        tokenized, _, _ = self.st.tokenize_sentences(texts)
        with torch.no_grad():
            prob = self.tmoji(tokenized)
        return torch.tensor(prob)

    def get_one_sentence_vector(self, m, text):
        result = [m[t] if t in m else np.zeros(m.vector_size) for t in word_tokenize(text)]
        return torch.tensor(result, dtype=float)

    def get_gensim_word_vectors(self, texts: List[str]):
        n_tokens_in = self.n_tokens_in
        result = []
        for m in self.gensim:
            r = stack_and_pad_tensors([self.get_one_sentence_vector(m, text) for text in texts], n_tokens_in)
            result.append(r)
        result = [r.float() for r in result]
        result = torch.cat(result, 2)
        result = self.gensim_nn(result)
        return result

    def get_nltk_vectors(self, texts: List[str]):
        # https://gist.github.com/japerk/1909413
        from textblob import TextBlob
        sid = self.nltk_sid
        vsid = self.vader_sid
        pdict = self.pdict
        n_tokens_in = self.n_tokens_in
        rake = self.rake_nltk
        nltk_texts = [word_tokenize(text) for text in texts]
        textblob_sentiments = [[sentiment.polarity, sentiment.subjectivity] for sentiment in [TextBlob(text).sentiment for text in texts]]
        textblob_sentiments = torch.tensor(textblob_sentiments).unsqueeze(1).expand(len(texts), n_tokens_in, 2)

        mask = stack_and_pad_tensors(list(map(lambda x: torch.ones(len(x), dtype=int), nltk_texts)), n_tokens_in)
        mask = self.is_mask_em(mask)
        has_digit = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([has_digits(str(t)) for t in x]), nltk_texts)), n_tokens_in)
        has_digit = self.has_digit_em(has_digit)

        m = self.text_model
        nltk_emb = stack_and_pad_tensors([torch.tensor([m[t] for t in sent]) for sent in nltk_texts], n_tokens_in) # if t in m else np.zeros(m.vector_size)
        sid_vec = torch.tensor([list(sid.polarity_scores(t).values()) for t in texts])
        sid_vec = sid_vec.unsqueeze(1).expand(len(texts), n_tokens_in, sid_vec.size(1))
        vsid_vec = torch.tensor([list(vsid.polarity_scores(t).values()) for t in texts])
        vsid_vec = vsid_vec.unsqueeze(1).expand(len(texts), n_tokens_in, vsid_vec.size(1))
        conlltags = [[ptags for ptags in nltk.tree2conlltags(ne_chunk(pos_tag(x)))] for x in nltk_texts]

        pos = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[tag.lower()] for token, tag, ne in x]), conlltags)), n_tokens_in)
        pos_emb = self.tag_em(pos)
        ner = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[ne.lower().split("-")[-1]] for token, tag, ne in x]), conlltags)), n_tokens_in)
        ner_emb = self.tag_em(ner)

        phrases = [get_rake_nltk_phrases(rake, t) for t in texts]

        key_wc_rake_nltk = [get_rake_nltk_wc(tokens, phr) for tokens, phr in zip(nltk_texts, phrases)]
        key_wc_rake_nltk = stack_and_pad_tensors(key_wc_rake_nltk, self.n_tokens_in)
        nltk_rake_vectors = self.key_wc_rake_nltk(key_wc_rake_nltk)

        result = torch.cat([vsid_vec, nltk_emb, textblob_sentiments, pos_emb, ner_emb, nltk_rake_vectors, sid_vec, mask, has_digit], 2)
        result = self.nltk_nn(result)
        return result

    def get_sentence_vector(self, texts: List[str]):
        tm = self.text_model
        n_tokens_in = self.n_tokens_in
        result = torch.tensor([tm.get_sentence_vector(text) for text in texts])
        result = self.full_sent_nn(result)
        result = result.unsqueeze(1).expand(len(texts), n_tokens_in, result.size(1))
        return result

    def get_stanford_nlp_vectors(self, texts: List[str]):
        snlp = self.snlp
        pdict = self.pdict
        n_tokens_in = self.n_tokens_in
        docs = [list(map(lambda x: dict(**x.to_dict()[0], ner=x.ner), snlp(doc).iter_tokens())) for doc in texts]

        upos = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token["upos"].lower()] for token in x]), docs)), n_tokens_in)
        upos_emb = self.tag_em(upos)

        xpos = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token["xpos"].lower()] for token in x]), docs)), n_tokens_in)
        xpos_emb = self.tag_em(xpos)

        deprel = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token["deprel"].split(":")[0].lower()] for token in x]), docs)),
            n_tokens_in)
        deprel_emb = self.tag_em(deprel)

        deprel2 = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor(
                [pdict[token["deprel"].split(":")[1].lower()] if ":" in token["deprel"] else 0 for token in x]), docs)),
            n_tokens_in)
        deprel_emb2 = self.tag_em(deprel2)

        sner = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor(
                [pdict[token["ner"].split("-")[1].lower()] if "-" in token["ner"] else 0 for token in x]), docs)),
            n_tokens_in)
        sner_emb = self.tag_em(sner)

        result = torch.cat(
            [upos_emb, xpos_emb, deprel_emb, sner_emb, deprel_emb2], 2)
        result = self.snlp_nn(result)
        return result

    def get_spacy_nlp_vectors(self, texts: List[str]):
        pdict = self.pdict
        nlp = self.nlp
        n_tokens_in = self.n_tokens_in
        with torch.no_grad():
            spacy_texts = list(nlp.pipe(texts, n_process=1))
            text_tensors = list(map(lambda x: torch.tensor(x.tensor), spacy_texts))
            text_tensors = stack_and_pad_tensors(text_tensors, n_tokens_in)
            head_tensors = stack_and_pad_tensors(list(map(lambda x: torch.tensor([t.head.tensor for t in x]), spacy_texts)), n_tokens_in)
        wl = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([len(token) - 1 for token in x]).clamp(0, 15), spacy_texts)), n_tokens_in)
        wl_emb = self.w_len(wl)
        wc = (torch.tensor(list(map(len, spacy_texts))) / 10).long().unsqueeze(1).expand(len(texts), n_tokens_in)
        wc_emb = self.wc_emb(wc)

        mask = stack_and_pad_tensors(list(map(lambda x: torch.ones(len(x), dtype=int), spacy_texts)), n_tokens_in)
        mask = self.is_mask_em(mask)
        has_digit = stack_and_pad_tensors(list(map(lambda x: torch.tensor([has_digits(str(t)) for t in x]), spacy_texts)), n_tokens_in)
        has_digit = self.has_digit_em(has_digit)

        pos = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token.pos_.lower()] for token in x]), spacy_texts)), n_tokens_in)
        pos_emb = self.tag_em(pos)
        tag = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token.tag_.lower()] for token in x]), spacy_texts)), n_tokens_in)
        tag_emb = self.tag_em(tag)
        dep = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token.dep_.lower()] for token in x]), spacy_texts)), n_tokens_in)
        dep_emb = self.tag_em(dep)
        sw = stack_and_pad_tensors(list(map(lambda x: torch.tensor([int(token.is_stop) for token in x]), spacy_texts)),
                                   n_tokens_in)
        sw_emb = self.sw_em(sw)
        ner = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token.ent_type_.lower()] for token in x]), spacy_texts)), n_tokens_in)
        ner_emb = self.tag_em(ner)

        is_oov = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([int(token.is_oov) for token in x]), spacy_texts)),
            n_tokens_in)
        is_oov_em = self.is_oov_em(is_oov)

        sent_start = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([int(token.sent_start) for token in x]), spacy_texts)),
            n_tokens_in)
        sent_start_em = self.sent_start_em(sent_start)

        head_dist = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([float(token.idx - token.head.idx) for token in x]), spacy_texts)),
            n_tokens_in)
        head_dist = head_dist.unsqueeze(2).expand(len(texts), n_tokens_in, 2)

        result = torch.cat(
            [text_tensors, pos_emb, tag_emb, dep_emb, sw_emb, ner_emb, wl_emb,
             wc_emb, mask, has_digit, is_oov_em, sent_start_em, head_dist, head_tensors], 2)
        result = self.spacy_nn(result)
        return result, spacy_texts

    def get_ibm_max(self, texts: List[str]):
        with torch.no_grad():
            result = self.ibm_max.predict(texts)
        result = self.ibm_nn(result)
        result = result.unsqueeze(1).expand(len(texts), self.n_tokens_in, result.size(1))
        return result

    def get_tmoji(self, texts: List[str]):
        with torch.no_grad():
            tm_probas = self.get_torchmoji_probas(texts)
        tm_probas = self.tm_nn(tm_probas)
        tm_probas = tm_probas.unsqueeze(1).expand(len(texts), self.n_tokens_in, tm_probas.size(1))
        return tm_probas

    def get_keyphrases(self, texts: List[str], spacy_texts):
        tm = self.text_model
        results = [get_pytextrank_wc_keylen(i) for i in spacy_texts]
        key_wc_pytextrank, key_occ_cnt_pytextrank = zip(*results)
        key_wc_pytextrank = stack_and_pad_tensors(key_wc_pytextrank, self.n_tokens_in)
        key_occ_cnt_pytextrank = stack_and_pad_tensors(key_occ_cnt_pytextrank, self.n_tokens_in)
        pytextrank_vectors = torch.cat((self.key_wc_pytextrank(key_wc_pytextrank), self.key_occ_cnt_pytextrank(key_occ_cnt_pytextrank)), 2) # 16

        yake_ke = self.kw_extractor
        yake_embs = [[tm.get_sentence_vector(s) for s in map(itemgetter(0), yake_ke.extract_keywords(t))] if has_words(t) else [np.zeros(300)] for t in texts]
        yake_embs = torch.tensor([np.average(yk, axis=0, weights=softmax(list(range(len(yk), 0, -1)))).astype(np.float32) if len(yk) > 0 else np.zeros(tm.get_dimension(), dtype=np.float32) for yk in yake_embs])
        yake_embs = self.yake_nn(yake_embs).unsqueeze(1).expand(len(texts), self.n_tokens_in, self.yake_dims)

        rake_ke = self.rake
        rake_embs = [[tm.get_sentence_vector(s) for s in map(itemgetter(0), rake_ke.apply(t))] if has_words(t) else [np.zeros(300)] for
                     t in texts]
        rake_embs = torch.tensor(
            [np.average(rk, axis=0, weights=softmax(list(range(len(rk), 0, -1)))).astype(np.float32) if len(rk) > 0 else np.zeros(tm.get_dimension(), dtype=np.float32) for rk in rake_embs])
        rake_embs = self.rake_nn(rake_embs).unsqueeze(1).expand(len(texts), self.n_tokens_in, self.rake_dims)

        result = torch.cat([pytextrank_vectors, yake_embs, rake_embs], 2)
        result = self.keyphrase_nn(result)
        return result

    def get_word_vectors(self, texts: List[str]):
        cap_method = {"snlp": self.get_stanford_nlp_vectors, "full_view": self.get_sentence_vector,
                      "nltk": self.get_nltk_vectors,
                      "ibm_max": self.get_ibm_max, "tmoji": self.get_tmoji, "gensim": self.get_gensim_word_vectors,
                      "fasttext_crawl": self.get__crawl_word_vectors}
        results = []
        if "spacy" in self.capabilities:
            r, spt = self.get_spacy_nlp_vectors(texts)
            results.append(r)
        if "key_phrases" in self.capabilities and "spacy" in self.capabilities:
            r = self.get_keyphrases(texts, spt)
            results.append(r)
        for c in self.capabilities:
            if c == "spacy" or c == "key_phrases":
                continue
            r = cap_method[c](texts)
            results.append(r)

        result = torch.cat(results, 2)
        result = self.contract_nn(result)
        return result
