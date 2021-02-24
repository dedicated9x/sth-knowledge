import pyautogui
import time
import pathlib as pl
import mini_projects.get_klause_vocab.lib.keyboard_developer as keyboard_developer
from mini_projects.get_klause_vocab.lib.config import PATH_TO_KLAUS
from mini_projects.get_klause_vocab.klaus.app import AppInfo

# for i in range(10):
#     print(pyautogui.position())
#     time.sleep(1)


#TODO dwa clickery dziedziczace
#TODO KlausApp
class Clicker:
    def __init__(self):
        location_set = AppInfo.get_location_set()
        self.form_pos, self.buttons_pos = location_set.explore_db_form, location_set.explore_db_button
        keyboard_developer.add_german_vk_codes()

    def _prepare(self):
        pyautogui.click(self.form_pos)
        pyautogui.moveTo(self.buttons_pos)

    def _enter_word(self, word):
        pyautogui.write(word)
        time.sleep(0.1)
        pyautogui.press('enter')
        pyautogui.click()
        pyautogui.write(['backspace'] * (len(word) + 1))

    def enter_words(self, words):
        self._prepare()
        for word in words:
            self._enter_word(word)

    # TODO
    def enter_words_from_file(self, exercise_fname):
        with open(pl.Path(PATH_TO_KLAUS).joinpath('exercises', exercise_fname), 'r') as infile:
            words = infile.read().splitlines()
        print(len(words))
        self.enter_words(words)


""" test """
from mini_projects.get_klause_vocab.test.factories import get_sample_fullwords
words = get_sample_fullwords(20)
ck = Clicker()
ck.enter_words(words[:20])


""" MAIN """
# 'links_1_part_0_of_42.txt'
# Clicker().enter_words_from_file(exercise_fname)


""" test bazy danych """
# import itertools
# from mini_projects.get_klause_vocab.klaus.basic_db import BasicDB
# idioms_to_fullwords = BasicDB.get_idioms_to_fullwords()
# fullwords = list(itertools.chain(*[idioms_to_fullwords[k] for k in idioms_to_fullwords.keys()]))
#
# assert len(idioms_to_fullwords) == 9318
# assert len(fullwords) == 10714


