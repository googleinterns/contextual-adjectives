"""Contains function for calculating BERT embeddings"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from scipy.spatial.distance import cosine, euclidean


class BertEmbedding(object):
    """Class for calculating embeddings between two texts"""
    def __init__(self, bert_model='bert-base-uncased', max_seq_length=50, device='cpu'):
        """Initializing the BERT model"""
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length

        self.device = torch.device("cpu" if device=='cpu' or not torch.cuda.is_available() else "cuda")
        n_gpu = torch.cuda.device_count()

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)

        self.model = BertModel.from_pretrained(self.bert_model)
        self.model.to(self.device)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()


    def get_embeddings(self, sentences, layer=-1):
        """Returns embeddings of words/sentences"""
        assert isinstance(sentences, list)
        for pair in sentences:
            assert len(pair) == 1
        examples = self._read_examples(sentences)

        features = self._convert_examples_to_features(
            examples=examples)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=16)

        out_features = []

        for input_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

            all_encoder_layers, _ = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            values = torch.mean(all_encoder_layers[layer], 1)
                
            out_features.append(values.detach().cpu().numpy())

        flat_list = [item for sublist in out_features for item in sublist]
        return flat_list

    def _convert_examples_to_features(self, examples):
        """Generate features of examples"""
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text)

            
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0:(self.max_seq_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(input_type_ids) == self.max_seq_length

            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features


    def _read_examples(self, inp):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        unique_id = 0
        for a, in inp:
            line_a = a.strip()
            examples.append(
                InputExample(unique_id=unique_id, text=line_a))
            unique_id += 1
        return examples
                        


class InputExample(object):
    """Input an example"""
    def __init__(self, unique_id, text):
        self.unique_id = unique_id
        self.text = text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
