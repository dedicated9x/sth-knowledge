import pyautogui
import time
from screeninfo import get_monitors
from dataclasses import dataclass
import mini_projects.get_klause_vocab.lib.keyboard_developer as keyboard_developer
from mini_projects.get_klause_vocab.config import PATH_TO_KLAUS_DIR
import pathlib as pl
from mini_projects.claus.lib.text_to_wav_converter import KlausTextToWavConverter
from mini_projects.claus.lib.clipboard_controller import ClipboardController
from mini_projects.get_klause_vocab.lib.keyboard_layout import KeyboardLayout
# for i in range(10):
#     print(pyautogui.position())
#     time.sleep(1)


@dataclass
class LocationSet:
    explore_db_form: tuple
    explore_db_button: tuple
    add_pl: tuple
    add_de: tuple
    add_comment: tuple
    add_record: tuple

class AppInfo:
    @staticmethod
    def get_location_set():
        monitors = get_monitors()
        if len(monitors) == 3:
            return LocationSet(
                explore_db_form=(822, 452),
                explore_db_button=(1382, 706),
                add_pl=(1207, 638),
                add_de=(1207, 745),
                add_comment=(1207, 973),
                add_record=(1253, 808)
            )
        else:
            return LocationSet(
                explore_db_form=(392, 201),
                explore_db_button=(1087, 520),
                add_pl=None,
                add_de=None,
                add_comment=None,
                add_record=None
            )


class AppView:
    def __init__(self):
        keyboard_developer.add_german_vk_codes()
        self.location_set = AppInfo.get_location_set()


class ExploreDatabaseView(AppView):
    def __init__(self):
        super(ExploreDatabaseView, self).__init__()
        self.form_pos, self.buttons_pos = self.location_set.explore_db_form, self.location_set.explore_db_button

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

# TODO test starego po zmianach
# TODO probna baza (do static)


class AddWordView(AppView):
    def __init__(self):
        super(AddWordView, self).__init__()
        self.pl_form_pos, self.de_form_pos, self.comment_form_pos, self.load_record_button = \
            self.location_set.add_pl, self.location_set.add_de, self.location_set.add_comment, self.location_set.add_record


pl2eng_transtab = str.maketrans("ąęóćźżńłśĆŹŻŃŁŚ", "aeoczznlsCZZNLS")


def add_word(self, word_pl, word_de, factor=1.0):
    global pl2eng_transtab

    pyautogui.click(self.pl_form_pos)
    pyautogui.write(word_pl.translate(pl2eng_transtab))

    KeyboardLayout.to_german()
    pyautogui.click(self.de_form_pos)
    pyautogui.write(word_de)
    KeyboardLayout.to_polish()

    # path_to_klaus_dir = pl.Path('E:\MedKlaus\Profesor Klaus 6.0 S³ownictwo')
    path_to_wav = KlausTextToWavConverter(pl.Path(PATH_TO_KLAUS_DIR)).convert(word_de.rstrip('-'), verbose=0)
    ClipboardController.save_to_clipboard(str(path_to_wav))

    pyautogui.click(self.load_record_button)
    time.sleep(0.2)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.25)
    pyautogui.press('enter')
    time.sleep(0.35)
    pyautogui.press('enter')
    time.sleep(0.35)
    pyautogui.press('enter')

"""part1"""
# from mini_projects.get_klause_vocab.test.factories import get_pt1
# pt1 = get_pt1()
#
# ct = 1
# self_ = AddWordView()
# for row in pt1.iterrows():
#     word_de = row[1]['de']
#     word_pl = row[1]['pl']
#     add_word(self_, word_pl, word_de)
#
#     ct += 1
#     print(ct)



"""correctness tests"""
# words_de = ["mittlere, Mittel-", "die Lendenwirbelsäule, die LWS", "die Mittelohrentzündung, die MOE", "die Hand", 'die Blinddarmentzündung, die Wurmfortsatzentzündung, die Appendizitis']
# words_pl = ["środkowy", "odcinek lędźwiowy kręgosłupa (2)", "zapalenie ucha środkowego (2)", "ręka", "zapalenie wyrostka robaczkowego (3)"]
#
# self_ = AddWordView()
# for word_pl, word_de in zip(words_pl, words_de):
#     add_word(self_, word_pl, word_de)

"""performance tests"""
# rg = list(range(1, 10))
#
# pol = [f"Polskie Slowko {i}" for i in rg]
# de = [f"Niemiecke Slowko {i}" for i in rg]
# fac = [1+0.2*i for i in rg]
#
# self_ = AddWordView()
#
# for word_pl, word_de, factor in zip(pol, de, fac):
#     add_word(self_, word_pl, word_de, factor)



# TODO przelaczenie klawiatury

# TODO baza danych od Natalii sparsowana i zrobiona
