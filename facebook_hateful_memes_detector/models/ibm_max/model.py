# the metadata of the model
# https://github.com/IBM/MAX-Toxic-Comment-Classifier
model_meta = {
    'id': 'max-toxic-comment-classifier',
    'name': 'MAX Toxic Comment Classifier',
    'description': 'BERT Base finetuned on toxic comments from Wikipedia.',
    'type': 'Text Classification',
    'source': 'https://developer.ibm.com/exchanges/models/all/max-toxic-comment-classifier/',
    'license': 'Apache V2'
}

import torch
from torch.nn import BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    for example in examples:
        tokens_a = tokenizer.tokenize(str(example))

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


import os
import logging
MODEL_NAME = 'BERT_PyTorch'
DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_PATH = f'{DIR}/assets/{MODEL_NAME}/'

# the output labels
LABEL_LIST = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

import torch
import time
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

logger = logging.getLogger()
import os


class ModelWrapper():
    def __init__(self):
        """Instantiate the BERT model."""

        # Load the model
        # 1. set the appropriate parameters
        self.eval_batch_size = 16
        self.max_seq_length = 64
        self.do_lower_case = True

        # 2. Initialize the PyTorch model
        model_state_dict = torch.load(DEFAULT_MODEL_PATH+'pytorch_model.bin', map_location='cpu')
        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_PATH, do_lower_case=self.do_lower_case)
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH,
                                                                             num_labels=len(LABEL_LIST),
                                                                             state_dict=model_state_dict)
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # 3. Set the layers to evaluation mode
        self.model.eval()

    def _pre_process(self, input):
        # Converting the input to features

        all_input_ids, all_input_mask, all_segment_ids = convert_examples_to_features(input, self.max_seq_length, self.tokenizer)
        # Turn input examples into batches
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.eval_batch_size)
        return test_dataloader

    def _post_process(self, result):
        """Convert the prediction output to the expected output."""
        # Generate the output format for every input string
        output = [{LABEL_LIST[0]: p[0],
                   LABEL_LIST[1]: p[1],
                   LABEL_LIST[2]: p[2],
                   LABEL_LIST[3]: p[3],
                   LABEL_LIST[4]: p[4],
                   LABEL_LIST[5]: p[5],
                   } for p in result]

        return output

    def _predict(self, test_dataloader):
        """Predict the class probabilities using the BERT model."""

        all_logits = []

        for step, batch in enumerate(test_dataloader):
            input_ids, input_mask, segment_ids = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            # Compute the logits
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()

            # Save the logits
            all_logits.append(logits.detach())
        all_logits = torch.cat(all_logits, axis=0).detach()
        # Return the predictions
        return all_logits

    def predict(self, x):
        pre_x = self._pre_process(x)
        prediction = self._predict(pre_x)
        # result = self._post_process(prediction)
        return prediction

import time
if __name__=="__main__":
    TEST_SENTENCES = ['I love mom\'s cooking',
                      'I love how you never reply back..',
                      'I love cruising with my homies',
                      'I love messing with yo mind!!',
                      'I love you and now you\'re just gone..',
                      'This is shit',
                      'This is the shit',
                      'Those people did not do a good job, such stupor.',
                      'Yeah and you think you can dance, everyone believes they can dance these days',
                      'Pets, pestering like creatures running for your attention, gotta love\'em.',
                      "I would like to punch you.",
                      "In hindsight, I do apologize for my previous statement."]
    model = ModelWrapper()
    from more_itertools import flatten
    print(model.predict(TEST_SENTENCES))
    import random
    big_list = list(flatten([list(map(lambda x: x + str(random.randint(0, 1e6)), TEST_SENTENCES)) for i in range(50)]))
    start = time.time()
    # Test 10 times
    _ = [model.predict(big_list) for _ in range(10)]
    end = time.time() - start
    print("Time taken = ", (end/10), "For list size = ", len(big_list))



