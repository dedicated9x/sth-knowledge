import pyautogui
import time
from screeninfo import get_monitors
import mini_projects.get_klause_vocab.keyboard_developer as keyboard_developer

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

""" MAIN """
ck = Clicker()
ck._prepare()
pyautogui.write('a ßüöä ÜÖÄ a', interval=0.25)


