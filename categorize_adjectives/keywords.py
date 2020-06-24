"""Categorize Adjectives by looking for keywords in definition"""
import os
from nltk.corpus import wordnet as wn

def find_adjectives(adjectives, categories, keywords_dictionary):
    """For each adjective, finding similar adjectives and adding to respective categories

    Create a small cluster of similar adjectives using similar_tos and also add antonyms to it.

    Arguments
    adjectives: A synset style list of adjectives
    categories: A dictionary of category_name to adjective clusters
    keywords_dictionary: A dictionary of category name to keywords being used to find adjectives
    for the given category

    Return: Categories with new adjectives as per keywords added.
    """
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
                similar_adj.append(lemma.name().lower())
                if lemma.antonyms():
                    similar_adj.append(lemma.antonyms()[0].name())
        # Finding Categories to which all these adjs should belong
        selected_cat = []
        for category, category_keywords in keywords_dictionary.items():
            for keyword in category_keywords:
                if keyword in definition:
                    selected_cat.append(category)

        for category in selected_cat:
            categories[category].extend(similar_adj)

    # Removing duplicates from categories
    comp_set = set()
    for category in categories:
        categories[category] = list(set(categories[category]))
        comp_set = comp_set | set(categories[category])
    return categories

def save_to_csv(categories, file_name):
    """Saving categories in a CSV file (file_name)"""
    file = open(file_name, "w")
    for category, adjectives in categories.items():
        sentence = category + ", "
        for adj in adjectives:
            sentence += adj + ","
        sentence = sentence[:-1]
        sentence += "\n"
        file.write(sentence)
    file.close()

if __name__ == "__main__":
    # Folder where generated files are stored
    generated_file = os.path.join(os.getcwd(), '..', 'generated_files/')

    # Fetching all adjective synsets from Wordnet
    all_adjectives = wn.all_synsets('a')

    # Semantic Categories for adjectives
    initial_categories = {'Perceptional' : [], 'Spatial' : [], 'Time' : [], 'Motion' : [],
                          'Substance' : [], 'Weather' : [], 'Body' : [], 'Mood' : [],
                          'Spirit' : [], 'Behavior' : [], 'Social' : [], 'Quantity' : [],
                          'Relational' : [], 'General' : [], 'Others' : []}

    # Keywords used to lookout for adjectives
    # Categories and keywords taken from
    # http://www.sfs.uni-tuebingen.de/projects/ascl/GermaNet/adjectives.shtml
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
    keywords['Behavior'] = ['behavior', 'character', 'animal specific', 'skill', 'relation',
                            'sympathy', 'inclination']
    keywords['Social'] = ['stratum', 'social', 'institution', 'poltics', 'religion', 'state',
                          'region']
    keywords['Quantity'] = ['quantity', 'number', 'cost', 'return']
    keywords['Relational'] = ['relational', 'validity', 'certainty', 'requirement',
                              'difficulty', 'energy requirement', 'functioning', 'security',
                              'linking', 'correspondence', 'accuracy', 'completeness', 'cause',
                              'reference', 'beneficial effect', 'effectiveness', 'order']
    keywords['General'] = ['comparative', 'evaluation', 'standard']

    categories_with_adjs = find_adjectives(all_adjectives, initial_categories, keywords)
    save_to_csv(categories_with_adjs, generated_file + "keyword_clusters.csv")
