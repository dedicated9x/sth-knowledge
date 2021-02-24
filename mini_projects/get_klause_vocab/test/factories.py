from mini_projects.get_klause_vocab.klaus.basic_db import BasicDB
import itertools
import random

# TODO (kiedyś) dwie tabele w pandasie i 'joiny' vs OOP

def get_sample_fullwords(lenght):
    def is_german_word(word):
        return (set(word) & {'ß', 'ö', 'ä'}) != set()

    idioms_to_fullwords = BasicDB.get_idioms_to_fullwords()
    german_words = list(filter(is_german_word, idioms_to_fullwords.keys()))
    sample_idioms = random.choices(german_words, k=lenght)
    # TODO to gdzies indziej powinno byc itertools
    sample_fullwords = list(itertools.chain(*[idioms_to_fullwords[k] for k in sample_idioms]))
    return sample_fullwords

# fullwords = get_sample_fullwords()



