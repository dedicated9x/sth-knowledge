from mini_projects.get_klause_vocab.lib.parsed_collections import idioms_to_fullwords
from mini_projects.get_klause_vocab.lib.config import PATH_TO_KLAUS
import requests
import pathlib as pl
import itertools


EXERCISE_SIZE = 20


class WordExtractor:
    def __init__(self):
        pass

    def _get_links(self, filename):
        with open(pl.Path(PATH_TO_KLAUS).joinpath('links', filename), 'r') as infile:
            return infile.read().splitlines()

    def _save_to_file(self, words, links_fname):
        global EXERCISE_SIZE
        l = EXERCISE_SIZE
        no_exercises = divmod(len(words), l)[0]
        chunks = [words[i * l: (i + 1) * l] for i in range(no_exercises - 1)]

        for n, words_exercise in enumerate(chunks):
            exercise_fname = f'links_{links_fname[:-4]}_part_{n}_of_{no_exercises}.txt'
            with open(pl.Path(PATH_TO_KLAUS).joinpath('exercises', exercise_fname), 'w') as outfile:
                outfile.write('\n'.join(words_exercise) + '\n')

    def extract_words(self, filename):
        links = self._get_links(filename)
        cumm_text = ''
        for elem in links:
            try:
                cumm_text += str(requests.get(elem).content)
            except requests.exceptions.ConnectionError:
                print(f'ERROR. {elem} nie dziala.')
        strings = cumm_text.lower().split(' ')
        idioms = set(idioms_to_fullwords.keys()) & set(strings)
        words = list(itertools.chain(*[idioms_to_fullwords[k] for k in idioms]))
        self._save_to_file(words, filename)



"""usage"""
# filename = '1.txt'
# WordExtractor().extract_words(filename)