from mini_projects.get_klause_vocab.parsed_collections import idioms_to_fullwords
import itertools
import random

# TODO (kiedyś) dwie tabele w pandasie i 'joiny' vs OOP

def get_sample_fullwords():
    def is_german_word(word):
        return (set(word) & {'ß', 'ö', 'ä'}) != set()

    german_words = list(filter(is_german_word, idioms_to_fullwords.keys()))
    sample_idioms = random.choices(german_words, k=20)
    sample_fullwords = list(itertools.chain(*[idioms_to_fullwords[k] for k in sample_idioms]))
    return sample_fullwords

fullwords = get_sample_fullwords()



