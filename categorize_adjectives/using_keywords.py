"""Categorize Adjectives by looking for keywords in definition"""
import os
from nltk.corpus import wordnet as wn

# Folder where generated files are stored
generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

# Fetching all adjective synsets from Wordnet
adjectives = wn.all_synsets('a')

# Semantic Categories for adjectives
categories = {'Perceptional' : [], 'Spatial' : [], 'Time' : [], 'Motion' : [], 'Substance' : [],
              'Weather' : [], 'Body' : [], 'Mood' : [], 'Spirit' : [], 'Behavior' : [],
              'Social' : [], 'Quantity' : [], 'Relational' : [], 'General' : [], 'Others' : []}

# Keywords used to lookout for adjectives
keywords = {}
keywords['Perceptional'] = ['perceptional', 'lightness', 'color', 'sound', 'taste', 'smell',
                            'surface']
keywords['Spatial'] = ['spatial', 'dimension', 'direction', 'localization', 'origin', 'form',
                       'existence']
keywords['Time'] = ['time', 'velocity', 'age', 'habit']
keywords['Motion'] = ['motion']
keywords['Substance'] = ['substance', 'composition', 'state', 'stability', 'consistency',
                         'ripeness', 'dampness', 'purity', 'gravity', 'weight', 'physics',
                         'chemistry', 'temperature']
keywords['Weather'] = ['weather', 'climate']
keywords['Body'] = ['body', 'life', 'constitution', 'affliction', 'desire', 'sex', 'appearance',
                    'bodily state']
keywords['Mood'] = ['feeling', 'stimulus', 'mood']
keywords['Spirit'] = ['spirit', 'intelligence', 'attention', 'knowledge', 'experience']
keywords['Behavior'] = ['behavior', 'character', 'animal specific', 'skill', 'relation', 'sympathy',
                        'inclination']
keywords['Social'] = ['stratum', 'social', 'institution', 'poltics', 'religion', 'state', 'region']
keywords['Quantity'] = ['quantity', 'number', 'cost', 'return']
keywords['Relational'] = ['relational', 'validity', 'certainty', 'requirement', 'effectiveness',
                          'difficulty', 'energy requirement', 'functioning', 'security', 'order',
                          'linking', 'correspondence', 'accuracy', 'completeness', 'cause',
                          'reference', 'beneficial effect']
keywords['General'] = ['comparative', 'evaluation', 'standard']

# For each adjective, finding similar adjectives and adding to respective categories
for adjective in adjectives:
    definition = adjective.definition().lower()
    definition = definition.split(" ")
    adj_name = adjective.lemmas()[0].name().lower()
    similar_synset = wn.synset(adjective.name()).similar_tos()
    similar_adj = []
    similar_adj.append(adj_name)
    # Finding similar adjectives
    for syn in similar_synset:
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            similar_adj.append(name)
            if lemma.antonyms():
                similar_adj.append(lemma.antonyms()[0].name())
    # Finding Categories to which all these adjs should belong
    selected_cat = []
    for category, category_keywords in keywords.items():
        for keyword in category_keywords:
            if keyword in definition:
                selected_cat.append(category)
                continue
    for category in selected_cat:
        categories[category].extend(similar_adj)

# Removing duplicates from categories
comp_set = set()
for category in categories:
    categories[category] = list(set(categories[category]))
    comp_set = comp_set | set(categories[category])

# Saving categories in a CSV file
f = open(generated_file + "keyword_clusters.csv", "w")
for category, adjectives in categories.items():
    sentence = category + ", "
    for adj in adjectives:
        sentence += adj + ","
    sentence = sentence[:-1]
    sentence += "\n"
    f.write(sentence)
f.close()
