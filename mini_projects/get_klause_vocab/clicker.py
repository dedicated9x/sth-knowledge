import pyautogui
import time
import pathlib as pl
from screeninfo import get_monitors
import mini_projects.get_klause_vocab.lib.keyboard_developer as keyboard_developer
from mini_projects.get_klause_vocab.lib.config import PATH_TO_KLAUS

# for i in range(10):
#     print(pyautogui.position())
#     time.sleep(1)

class Localizer:
    def __init__(self):
        monitors = get_monitors()
        if len(monitors) == 3:
            positions = (822, 452), (1382, 706)
        else:
            positions = (392, 201), (1087, 520)

        self.form_pos, self.button_pos = positions


class Clicker:
    def __init__(self):
        localizer = Localizer()
        self.form_pos, self.buttons_pos = localizer.form_pos, localizer.button_pos
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

    def enter_words_from_file(self, exercise_fname):
        with open(pl.Path(PATH_TO_KLAUS).joinpath('exercises', exercise_fname), 'r') as infile:
            words = infile.read().splitlines()
        print(len(words))
        self.enter_words(words)


""" test """
# from mini_projects.get_klause_vocab.test.fabrics import get_sample_fullwords
# words = get_sample_fullwords()
# ck = Clicker()
# ck.enter_words(words[:20])


""" MAIN """
# TODO clicker (pobiera 20 i wklikuje)
# exercise_fname = 'links_1_part_0_of_42.txt'
# Clicker().enter_words_from_file(exercise_fname)
