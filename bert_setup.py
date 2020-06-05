"""Contains class bert for operations related to BERT"""
from nltk.tokenize.treebank import TreebankWordTokenizer
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

class Bert:
    """Given an index in tokens of sentence, suggest new words with the BERT confidence score.

    Attributes:
        tokenizer : Tokenize a sentence using NLTK Tokenizer.
        bert_tokenizer : Tokenize a sentence using Bert_Tokenizer.
        bert_model : Predict new words for a token in the sentence.
    """
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()
        self.bert_tokenizer = Bert_Tokenizer.from_pretrained('bert-large-uncased')
        self.bert_model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.bert_model.eval()

    def perturb_bert(self, sentence, num, masked_index):
        """Return a list of similar words suggested by BERT with BERT scores

        Args:
        sentence : sentence which acts as a context for new words suggestion.
        num : Number of new words need to be suggested.
        masked_index : the index of the word in the sentence for which new
        words need to be suggested

        Returns:
        A dictionary with new words suggested as keys and their confidence scores
        as values.
        """
        tokens = self.tokenizer.tokenize(sentence)

        # for each idx, use Bert to generate k (i.e., num) candidate tokens
        original_word = tokens[masked_index]
        # Getting the base form of the word to check for it's synonyms
        low_tokens = [x.lower() for x in tokens]
        low_tokens[masked_index] = '[MASK]'
        # Eliminating cases for "'s" as Bert does not work well on these cases.
        if original_word == "'s":
            return {}
        # try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(low_tokens)
            prediction = self.bert_model(torch.tensor([indexed_tokens]))

        except KeyError:
            return {}

        # get the similar words
        topk_idx = torch.topk(prediction[0, masked_index], num)[1].tolist()
        scores = torch.topk(prediction[0, masked_index], num)[0].tolist()
        topk_tokens = self.bert_tokenizer.convert_ids_to_tokens(topk_idx)
        # generate similar sentences
        tokens[masked_index] = original_word
        token_score = {}
        for i, score in enumerate(scores):
            token_score[topk_tokens[i]] = score
        return token_score
