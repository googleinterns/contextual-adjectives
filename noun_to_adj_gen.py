import nltk
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import string
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from bert_embedding import BertEmbedding
import pickle
from nltk.corpus import wordnet
import tensorflow as tf

noun_list = []
for synset in list(wordnet.all_synsets(wordnet.NOUN)):
	noun = synset.lemmas()[0].name()
	if noun.isalpha():
		noun_list.append(noun.lower())

adj_list = []
for synset in list(wordnet.all_synsets(wordnet.ADJ)):
	adj_list.append(synset.lemmas()[0].name().lower())

noun_list = list(set(noun_list))
adj_list = list(set(adj_list))
noun_to_adj = {}
for noun in noun_list:
	noun_to_adj[noun] = []
sentences = []
with open("all.txt","r") as f:
	sentences = f.readlines()
for i in range(len(sentences)):
	sentences[i] = sentences[i].split("\n")[0]  


# Use nltk treebank tokenizer and detokenizer
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

# when we use Bert
berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
bertmodel.eval()


# Generate a list of similar words suggested by Bert
def perturbBert(sent, model, num, masked_index, tagFlag):
	tokens = tokenizer.tokenize(sent)
	invalidChars = set(string.punctuation)

	# for each idx, use Bert to generate k (i.e., num) candidate tokens
	original_word = tokens[masked_index]
	#Getting the base form of the word to check for it's synonyms
	low_tokens = [x.lower() for x in tokens]        
	low_tokens[masked_index] = '[MASK]'
	#Eliminating cases for "'s" as Bert does not work well on these cases.      
	if original_word=="'s":
		return {}
	# try whether all the tokens are in the vocabulary
	try:
		indexed_tokens = berttokenizer.convert_tokens_to_ids(low_tokens)
		tokens_tensor = torch.tensor([indexed_tokens])
		prediction = model(tokens_tensor)

	except KeyError as error:
		return {}

	# get the similar words
	topk_Idx = torch.topk(prediction[0, masked_index], num)[1].tolist()
	scores = torch.topk(prediction[0, masked_index], num)[0].tolist()
	topk_tokens = berttokenizer.convert_ids_to_tokens(topk_Idx)
	# generate similar sentences
	tokens[masked_index] = original_word
	token_score = {}
	for i in range(len(topk_tokens)):
		token_score[topk_tokens[i]] = scores[i]
	return token_score


#Number of perturbations you want to make for a word in a sentence
num_of_perturb = 10
adj_tags = ['JJ','JJR','JJS']
noun_tags = ['NN','NNS','NNP','NNPS']
adj_set = set(adj_list)
noun_set = set(noun_list)
for sent in sentences:
    #POS tagging the sentence
	tokens = tokenizer.tokenize(sent)
	pos_inf = nltk.tag.pos_tag(tokens)
	for idx in range(len(pos_inf)):
		word, tag = pos_inf[idx]
		word = word.lower()
		if tag in noun_tags:
			if word in noun_list:
				if idx!=0:
					word1, tag1 = pos_inf[idx-1]
					word1 = word1.lower()
					if tag1 in adj_tags:
						tagFlag = tag1[:2]
						token_score = perturbBert(sent, bertmodel, num_of_perturb, idx-1,tagFlag)
						new_words = list(token_score.keys())
						intersection = list(set(new_words) & adj_set)
						intersection = [(a, token_score[a]) for a in intersection]
						noun_to_adj[word].extend(intersection)
						if word1 in adj_list:
							tagFlag = tag[:2]
							token_score = perturbBert(sent, bertmodel, num_of_perturb, idx,tagFlag)
							new_words = list(token_score.keys())
							for n_word in new_words:
								if n_word in noun_list:
									noun_to_adj[n_word].append((word1, token_score[n_word]))
				if idx!=(len(pos_inf)-1):
					word1, tag1 = pos_inf[idx+1]
					word1 = word1.lower()
					if tag1 in adj_tags:
						tagFlag = tag1[:2]
						token_score = perturbBert(sent, bertmodel, num_of_perturb, idx+1,tagFlag)
						new_words = list(token_score.keys())
						intersection = list(set(new_words) & adj_set)
						intersection = [(a, token_score[a]) for a in intersection]
						noun_to_adj[word].extend(intersection)
						if word1 in adj_list:
							tagFlag = tag[:2]
							token_score = perturbBert(sent, bertmodel, num_of_perturb, idx,tagFlag)
							new_words = list(token_score.keys())
							for n_word in new_words:
								if n_word in noun_list:
									noun_to_adj[n_word].append((word1, token_score[n_word]))    


with open("noun_to_adj_score.dat", 'wb') as f:
	pickle.dump(noun_to_adj, f)    
