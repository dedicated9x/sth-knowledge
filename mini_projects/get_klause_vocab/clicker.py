
# # TODO wpisanie z pliku
# def enter_words_from_file(self, exercise_fname):
#     with open(pl.Path(PATH_TO_KLAUS).joinpath('exercises', exercise_fname), 'r') as infile:
#         words = infile.read().splitlines()
#     print(len(words))
#     self.enter_words(words)


""" TEST - wpisanie 20-u slowek """
# from mini_projects.get_klause_vocab.klaus.app import ExploreDatabaseView
# from mini_projects.get_klause_vocab.test.factories import get_sample_fullwords
# words = get_sample_fullwords(20)
# ck = ExploreDatabaseView()
# ck.enter_words(words[:20])


""" TEST - z pliku """
# 'links_1_part_0_of_42.txt'
# Clicker().enter_words_from_file(exercise_fname)


""" TEST - bazy danych """
import itertools
from mini_projects.get_klause_vocab.klaus.basic_db import BasicDB
idioms_to_fullwords = BasicDB.get_idioms_to_fullwords()
fullwords = list(itertools.chain(*[idioms_to_fullwords[k] for k in idioms_to_fullwords.keys()]))

assert len(idioms_to_fullwords) == 9318
assert len(fullwords) == 10714


