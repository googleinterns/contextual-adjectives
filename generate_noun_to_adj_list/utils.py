"""Some basic functions required by Contextual Adjectives"""
from nltk.corpus import wordnet

def sentence_list_gen(filename):
    """Read lines from a file and make a list of lines.

    Args:
      filename : The address of the file which is being read

    Returns:
      Returns a list of lines in the file.
    """
    sentences = []
    with open(filename, "r") as file: #add this to readme file
        sentences = file.readlines()
    for i, sentence in enumerate(sentences):
        sentences[i] = sentence.split("\n")[0]
    return sentences

def noun_list_gen():
    """Generate a list of nouns from nltk wordnet database."""
    noun_list = []
    for synset in list(wordnet.all_synsets(wordnet.NOUN)):
        noun = synset.lemmas()[0].name()
        if noun.isalpha():
            noun_list.append(noun.lower())
    return list(set(noun_list))

def adjective_list_gen():
    """Generate a list of adjectives from nltk wordnet database."""
    adj_list = []
    for synset in list(wordnet.all_synsets(wordnet.ADJ)):
        adj_list.append(synset.lemmas()[0].name().lower())
    return list(set(adj_list))

def filter_by_idf(noun_to_adj, adj_idf, threshold, max_adj):
    """Filter by IDF values, i.e. remove adjectives with IDF
    lessthan threshold and return at max max_adj adjectives
    for a noun.

    Args:
      noun_to_adj : Given dictionary of nouns to adjectives with BERT score.
      adj_idf : IDF values of adjectives.
      threshold : Threshold to filter by IDF.
      max_adj : Max number of adjectives for a noun.

    Returns:
      A Dictionary of noun to adj after filtering in sorted order of BERT score.
    """
    noun_to_adj_sort = {}
    for noun in noun_to_adj.keys():
        total_bert_score = {}
        total = {}
        for adj, score in noun_to_adj[noun]:
            try:
                total_bert_score[adj] += score
                total[adj] += 1
            except KeyError:
                total_bert_score[adj] = score
                total[adj] = 1
        average_bert_score = {}
        for adj, val in total_bert_score.items():
            average_bert_score[adj] = -1*val/total[adj]
        try:
            sorted_adjs = sorted(average_bert_score.items(), key=lambda kv: (kv[1], kv[0]))[:max_adj]
        except IndexError:
            sorted_adjs = sorted(average_bert_score.items(), key=lambda kv: (kv[1], kv[0]))
        sorted_lis = []
        for key, val in sorted_adjs:
            if adj_idf[key] >= threshold:
                sorted_lis.append((key, -1*val))
        noun_to_adj_sort[noun] = sorted_lis
    return noun_to_adj_sort

def save_as_csv(filename, noun_to_adj):
    """Save a dictionary noun_to_adj to a given filename as csv."""
    file = open(filename, "w")
    for noun in noun_to_adj.keys():
        if noun_to_adj[noun] != []:
            sentence = noun + "\t"
            adjs = noun_to_adj[noun]
            for adj, val in adjs:
                sentence += adj + "(" + str(round(val, 2)) + ")\t"
            sentence = sentence[:-1]
            sentence += "\n"
            file.write(sentence)
    file.close()
