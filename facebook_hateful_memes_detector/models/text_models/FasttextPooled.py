import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors
from ...preprocessing import clean_text


class FasttextPooledModel(nn.Module):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0,
                 n_tokens_in=64, n_tokens_out=16,
                use_as_super=False,
                 **kwargs):
        super(FasttextPooledModel, self).__init__()
        fasttext_file = kwargs["fasttext_file"] if "fasttext_file" in kwargs else None
        fasttext_model = kwargs["fasttext_model"] if "fasttext_model" in kwargs else None
        assert fasttext_file is not None or fasttext_model is not None or use_as_super
        if fasttext_file is not None:
            self.text_model = fasttext.load_model(fasttext_file)
        else:
            self.text_model = fasttext_model

        projection = [nn.Linear(500, classifer_dims * 2), nn.LeakyReLU(),
                      nn.Linear(classifer_dims * 2, classifer_dims)]
        init_fc(projection[0], "leaky_relu")
        init_fc(projection[2], "linear")
        self.projection = nn.Sequential(*projection)
        layers = [GaussianNoise(gaussian_noise), nn.Linear(classifer_dims, classifer_dims, bias=True),
                  nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(classifer_dims, num_classes)]
        self.bpe = BPEmb(dim=100)
        self.cngram = CharNGram()
        init_fc(layers[1], "leaky_relu")
        init_fc(layers[4], "linear")
        self.classifier = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.binary = num_classes == 2
        self.eps = 1e-7
        self.auc_loss = False
        self.n_tokens_in = n_tokens_in
        self.n_tokens_out = n_tokens_out

    def get_sentence_vector(self, texts: List[str]):
        tm = self.text_model
        bpe = self.bpe
        cngram = self.cngram
        result = torch.tensor([tm.get_sentence_vector(text) for text in texts])
        result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-5)  # Normalize in sentence dimension
        res2 = torch.stack([self.get_one_sentence_vector(bpe, text).mean(0) for text in texts])
        res2 = res2 / res2.norm(dim=1, keepdim=True).clamp(min=1e-5)
        res3 = torch.stack([cngram[text] for text in texts])
        res3 = res3 / res3.norm(dim=1, keepdim=True).clamp(min=1e-5)
        result = torch.cat([result, res2, res3], 1)
        result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return result

    def get_one_sentence_vector(self, tm, sentence):
        tokens = fasttext.tokenize(sentence)
        if isinstance(tm, fasttext.FastText._FastText):
            result = torch.tensor([tm[t] for t in tokens])
        elif isinstance(tm, torchnlp.word_to_vector.char_n_gram.CharNGram):
            result = torch.stack([tm[t] for t in tokens])
        else:
            result = tm[tokens]
        return result

    def get_word_vectors(self, texts: List[str]):

        # expected output # Bx64x512
        bpe = self.bpe
        cngram = self.cngram
        tm = self.text_model
        n_tokens_in = self.n_tokens_in
        result = stack_and_pad_tensors([self.get_one_sentence_vector(tm, text) for text in texts], n_tokens_in)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        res2 = stack_and_pad_tensors([self.get_one_sentence_vector(bpe, text) for text in texts], n_tokens_in)
        res2 = res2 / res2.norm(dim=2, keepdim=True).clamp(min=1e-5)
        res3 = stack_and_pad_tensors([self.get_one_sentence_vector(cngram, text) for text in texts], n_tokens_in)
        res3 = res3 / res3.norm(dim=2, keepdim=True).clamp(min=1e-5)
        result = torch.cat([result, res2, res3], 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result

    def forward(self, texts: List[str], img, labels):
        projections, vectors = self.__get_scores__(texts, img)
        logits = self.classifier(projections)
        loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
        preds = logits.max(dim=1).indices
        logits = torch.softmax(logits, dim=1)
        if self.binary and self.training and self.auc_loss:
                # Projection aug loss
                pos_projections = projections[labels == 1]
                neg_projections = projections[labels == 0]
                pos_randomized = pos_projections[torch.randperm(pos_projections.size(0))]
                neg_randomized = neg_projections[torch.randperm(neg_projections.size(0))]
                pos_cos = (F.cosine_similarity(pos_projections, pos_randomized) + 1)/2
                neg_cos = (F.cosine_similarity(neg_projections, neg_randomized) + 1)/2
                min_len = min(pos_projections.size(0), neg_projections.size(0))
                pos_neg_cos = (F.cosine_similarity(pos_projections[:min_len], neg_projections[:min_len]) + 1)/2

                pl = ((pos_cos - 1)**2).mean()
                nl = ((neg_cos - 1)**2).mean()
                pnl = (pos_neg_cos**2).mean()
                projection_loss = (pl + nl + pnl)/3.0

                # aucroc loss
                #
                probas = logits[:, 1]
                pos_probas = labels * probas
                neg_probas = (1-labels) * probas
                pos_probas = pos_probas[pos_probas > self.eps]
                pos_probas_min = pos_probas.min()
                neg_probas_max = neg_probas.max() # torch.topk(neg_probas, 5, 0).values
                loss_1 = F.relu(neg_probas - pos_probas_min).mean()
                loss_2 = F.relu(neg_probas_max - pos_probas).mean()
                auc_loss = (loss_1 + loss_2)/2

                loss = 1.0 * loss + 0.0 * auc_loss + 0.0 * projection_loss # 0.1, 0.9 weights

        return logits, preds, projections, vectors, loss

    def __get_scores__(self, texts: List[str], img):
        projections = self.projection(self.get_sentence_vector(texts))
        return projections, projections.unsqueeze(1)




