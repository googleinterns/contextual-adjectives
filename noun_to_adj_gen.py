"""Code to generate noun to adjective dictionary"""
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
from bert_setup import Bert

class NounToAdjGen:
    """Add adjectives for nouns in dictionary noun_to_adj.

    Attributes:
        noun_to_adj : Noun to adjective dictionary.
        tokenizer : An instance of nltk's tokenizer.
        bert_model : An instance of class bert.
        adj_tags : Tags of adjectives in nltk.
        noun_tags : Tags of nouns in nltk.
        noun_list : List of nouns that we are working on.
        adj_list : List of adjectives that we are working on.
    """
    def __init__(self, noun_list, adj_list):
        """Initializing noun to adjective dictionary."""
        self.noun_to_adj = {}
        for noun in noun_list:
            self.noun_to_adj[noun] = []
        # Use nltk treebank tokenizer
        self.tokenizer = TreebankWordTokenizer()
        # Initializing the bert class
        self.bert_model = Bert()
        # https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
        self.adj_tags = ['JJ', 'JJR', 'JJS']
        self.noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        self.noun_list = noun_list
        self.adj_list = adj_list

    def add_to_dictionary(self, sentences, num_of_perturb):
        """Add adjectives for nouns by perturbing sentence to noun_to_adj.

        Args:
        sentences : The list of sentences for which to look up for nouns and adjs.
        num_of_perturb : Number of perturbations you want to make for a word in a sentence
        """
        for sent in sentences:
            # Tokenizing and POS tagging the sentence
            pos_inf = nltk.tag.pos_tag(self.tokenizer.tokenize(sent))
            for idx, (word, tag) in enumerate(pos_inf):
                word = word.lower()
                if tag in self.noun_tags and word in self.noun_list:
                    valid_adj_index = []
                    if idx != 0:
                        valid_adj_index.append(idx-1)
                    if idx != (len(pos_inf)-1):
                        valid_adj_index.append(idx+1)
                    for adj_index in valid_adj_index:
                        word1, tag1 = pos_inf[adj_index]
                        word1 = word1.lower()
                        if tag1 in self.adj_tags and word1 in self.adj_list:
                            self.add_adjectives(sent, num_of_perturb, adj_index, word)
                            self.add_nouns(sent, num_of_perturb, idx, word1)
                        elif tag1 in self.adj_tags:
                            self.add_adjectives(sent, num_of_perturb, adj_index, word)

    def add_adjectives(self, sent, num_of_perturb, adj_index, word):
        """Ask bert for suggestions for more adjectives and add their intersection
        with adjectives list to the dictionary.
        Args:
        sent : The sentence for which use bert to find more adjectives.
        num_of_perturb : Number of perturbations you want to make for a word in a sentence
        adj_index : The index of the word need to be perturbed in the sentence.
        word : The noun for which we are looking for adjectives
        """
        token_score = self.bert_model.perturb_bert(sent, num_of_perturb, adj_index)
        new_words = list(token_score.keys())
        intersection = list(set(new_words) & set(self.adj_list))
        intersection = [(a, token_score[a]) for a in intersection]
        self.noun_to_adj[word].extend(intersection)

    def add_nouns(self, sent, num_of_perturb, noun_index, word):
        """Ask bert for suggestions for more nouns and add their intersection with nouns
        list to the dictionary.
        Args:
        sent : The sentence for which use bert to find more adjectives.
        num_of_perturb : Number of perturbations you want to make for a word in a sentence
        adj_index : The index of the word need to be perturbed in the sentence.
        word : The noun for which we are looking for adjectives
        """
        token_score = self.bert_model.perturb_bert(sent, num_of_perturb, noun_index)
        new_words = list(token_score.keys())
        for n_word in new_words:
            if n_word in self.noun_list:
                self.noun_to_adj[n_word].append((word, token_score[n_word]))
    