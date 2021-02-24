import pyautogui
import time
from screeninfo import get_monitors
from dataclasses import dataclass
import mini_projects.get_klause_vocab.lib.keyboard_developer as keyboard_developer

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


word_pl = "Polskie Slowko"
word_de = "Niemiecke Slowko"
word_com = "komentarz"

self = AddWordView()

pyautogui.click(self.pl_form_pos)
pyautogui.write(word_pl)
pyautogui.click(self.de_form_pos)
pyautogui.write(word_de)
pyautogui.click(self.comment_form_pos)
pyautogui.write(word_com)


# TODO 01_here do wyjebania
# from mini_projects.claus.0
# path_to_wav = KlausTextToWavConverter.convert(text, verbose=1)



