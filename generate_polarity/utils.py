"""Some basic functions required by COntextual Adjectives"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_calculator(noun_to_adj):
    """Calculate Sentiment of adjectives wrt noun and save them in the noun_to_adj dictionary."""
    sid = SentimentIntensityAnalyzer()
    for noun, adjs in noun_to_adj.items():
        if adjs!=[]:
            polar = {}
            for adj, score in adjs:
                sentence = adj + " " + noun
                ss = sid.polarity_scores(sentence)['compound']
                polar[adj] = ss
                sorted_adjs = sorted(polar.items(), key = lambda kv:(kv[1], kv[0]))
                sorted_lis = []
                for key, val in sorted_adjs:
                    sorted_lis.append((key, val))
                noun_to_adj[noun] = sorted_lis
    return noun_to_adj            

def save_as_csv(filename, noun_to_adj):
    """Save a dictionary noun_to_adj to a given filename as csv."""
    file = open(filename, "w")
    for noun in noun_to_adj.keys():
        if noun_to_adj[noun] != []:
            sent = noun + "\t"
            adjs = noun_to_adj[noun]
            for adj, val in adjs:
                sent += adj + "(" + str(round(val, 2)) + ")\t"
            sent = sent[:-1]
            sent += "\n"
            file.write(sent)
    file.close()
