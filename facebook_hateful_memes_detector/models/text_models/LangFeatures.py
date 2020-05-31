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
import json
import csv
import numpy as np
import yake

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import stanza


import spacy

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_pos_tag_indices, pad_tensor, \
    get_penn_treebank_pos_tag_indices, get_all_tags
from ...utils import get_universal_deps_indices
from .FasttextPooled import FasttextPooledModel
from ..external import ModelWrapper
from ...utils import WordChannelReducer
from ..classifiers import CNN1DClassifier, GRUClassifier
from .Fasttext1DCNN import Fasttext1DCNNModel
import pytextrank



class LangFeaturesModel(Fasttext1DCNNModel):
    def __init__(self, classifer_dims, num_classes, embedding_dims,
                 gaussian_noise=0.0, dropout=0.0,
                 internal_dims=512, n_layers=2,
                 classifier="cnn",
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False,
                 **kwargs):
        super(LangFeaturesModel, self).__init__(classifer_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                                internal_dims, n_layers,
                                                classifier,
                                                n_tokens_in, n_tokens_out, True, **kwargs)
        capabilities = kwargs["capabilities"] if "capabilities" in kwargs else ["spacy"]
        self.capabilities = capabilities

        gru_layers = kwargs["gru_layers"] if "gru_layers" in kwargs else 2
        gru_dropout = kwargs["gru_dropout"] if "gru_dropout" in kwargs else 0.1
        tr = pytextrank.TextRank(token_lookback=7)
        self.nlp = spacy.load("en_core_web_lg", disable=[])
        self.nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        embedding_dim = 8
        if "snlp" in capabilities:
            self.snlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,ner', use_gpu=True,
                                    pos_batch_size=3000)
        if "key_phrases" in capabilities:
            self.kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                 dedupFunc=deduplication_algo, windowsSize=windowSize,
                                                 top=numOfKeywords, features=None)
            self.key_occ_cnt = nn.Embedding(8, embedding_dim)
            nn.init.normal_(self.key_occ_cnt.weight, std=1 / embedding_dim)
            self.key_wc = nn.Embedding(4, embedding_dim)
            nn.init.normal_(self.key_wc.weight, std=1 / embedding_dim)
        self.pdict = get_all_tags()

        cap_to_dim_map = {"spacy": embedding_dim * 7, "snlp": embedding_dim * 5, "tmoji": 64, "ibm_max": 7}
        all_dims = sum([cap_to_dim_map[c] for c in capabilities])
        self.all_dims = all_dims
        self.tag_em = nn.Embedding(len(self.pdict)+1, embedding_dim)
        # nn.init.normal_(self.embeds.weight, std=1 / embedding_dim)
        init_fc(self.tag_em, "linear")

        self.sw_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.sw_em.weight, std=1 / embedding_dim)

        self.w_len = nn.Embedding(16, embedding_dim)
        nn.init.normal_(self.w_len.weight, std=1 / embedding_dim)

        self.wc_emb = nn.Embedding(8, embedding_dim)
        nn.init.normal_(self.wc_emb.weight, std=1 / embedding_dim)
        if not use_as_super:
            if classifier == "cnn":
                self.classifier = CNN1DClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out, classifer_dims, internal_dims, None, gaussian_noise, dropout)
            elif classifier == "gru":
                self.classifier = GRUClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out, classifer_dims, internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

        gru_dims = kwargs["gru_dims"] if "gru_dims" in kwargs else int(classifer_dims / 2)
        if not use_as_super:
            self.projection = WordChannelReducer(gru_dims * 2, classifer_dims, 4)

        if "ibm_max" in capabilities:
            self.ibm_max = ModelWrapper()
        if "tmoji" in capabilities:
            with open(VOCAB_PATH, 'r') as f:
                maxlen = 64
                self.vocabulary = json.load(f)
                self.st = SentenceTokenizer(self.vocabulary, maxlen)
                self.tmoji = torchmoji_emojis(PRETRAINED_PATH)
        self.lstm = nn.Sequential(
            nn.GRU(all_dims, gru_dims, gru_layers, batch_first=True, bidirectional=True, dropout=gru_dropout))

    def get_torchmoji_probas(self,  texts: List[str]):
        tokenized, _, _ = self.st.tokenize_sentences(texts)
        prob = self.tmoji(tokenized)
        return torch.tensor(prob)

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
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result

    def get_spacy_nlp_vectors(self, texts: List[str]):
        pdict = self.pdict
        nlp = self.nlp
        n_tokens_in = self.n_tokens_in
        spacy_texts = list(nlp.pipe(texts, n_process=4))
        wl = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([len(token) - 1 for token in x]).clamp(0, 15), spacy_texts)), n_tokens_in)
        wl_emb = self.w_len(wl)

        if "key_phrases" in self.capabilities:
            wl = stack_and_pad_tensors(
                list(map(lambda x: torch.tensor([len(token) - 1 for token in x]).clamp(0, 15), spacy_texts)),
                n_tokens_in)
            wl_emb = self.w_len(wl)

        wc = (torch.tensor(list(map(len, spacy_texts))) / 10).long().unsqueeze(1).expand(len(texts), n_tokens_in)
        wc_emb = self.wc_emb(wc)

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
        result = torch.cat(
            [pos_emb, tag_emb, dep_emb, sw_emb, ner_emb, wl_emb,
             wc_emb], 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result, spacy_texts

    def get_ibm_max(self, texts: List[str]):
        result = self.ibm_max.predict(texts)
        result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-5)
        result = result.unsqueeze(1).expand(len(texts), self.n_tokens_in, result.size(1))
        return result

    def get_tmoji(self, texts: List[str]):
        tm_probas = self.get_torchmoji_probas(texts)
        tm_probas = tm_probas / tm_probas.norm(dim=1, keepdim=True).clamp(min=1e-5)
        tm_probas = tm_probas.unsqueeze(1).expand(len(texts), self.n_tokens_in, tm_probas.size(1))
        return tm_probas

    def get_keyphrases(self, texts: List[str], spacy_texts):
        pass

    def get_word_vectors(self, texts: List[str]):
        cap_method = {"snlp": self.get_stanford_nlp_vectors,
                      "ibm_max": self.get_ibm_max, "tmoji": self.get_tmoji}
        results = []
        if "spacy" in self.capabilities or "key_phrases" in self.capabilities:
            r, spt = self.get_spacy_nlp_vectors(texts)
            results.append(r)
        if "key_phrases" in self.capabilities:
            pass
        for c in self.capabilities:
            r = cap_method[c](texts)
            results.append(r)

        result = torch.cat(results, 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result
