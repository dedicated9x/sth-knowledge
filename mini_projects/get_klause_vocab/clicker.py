import pyautogui
import time
from screeninfo import get_monitors


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

# l = Localizer()
# x, y = l.button_pos
# Point(x=830, y=207)

# localizer = Localizer()
# form_pos, buttons_pos = localizer.form_pos, localizer.button_pos

class Clicker:
    def __init__(self):
        localizer = Localizer()
        self.form_pos, self.buttons_pos = localizer.form_pos, localizer.button_pos

    def _prepare(self):
        pyautogui.click(self.form_pos)
        pyautogui.moveTo(self.buttons_pos)

z1 = 'Hello äöüß ÄÖÜ world!'


letters_to_keys = {
    'ß': '-',
    'ü': '[',
    'ö': ';',
    'ä': "'",
    'Ü': '{',
    'Ö': ':',
    'Ä': '"',
}

def some_func(word):
    return ''.join([letter if letter.isascii() else letters_to_keys[letter] for letter in list(word)])

ck = Clicker()
# ck._prepare()
# pyautogui.write(some_func(z1), interval=0.25)

import ctypes
vkCode = 66
KEYEVENTF_KEYDOWN = 0
# ctypes.windll.user32.keybd_event(vkCode, 0, KEYEVENTF_KEYDOWN, 0)


import pyautogui._pyautogui_win as pyautogui_win
pyautogui_win.keyboardMapping.update({'ß': 0xDB})

""" dobry """
# ck._prepare()
time.sleep(1)

pyautogui.press('b')
pyautogui.press('ß')




# TODO 1 posprzataj i zrob aSSb do testowania
# TODO 2 dodaj pozostale litery

""" dobry - bezposrednio"""
# ck._prepare()
# ctypes.windll.user32.keybd_event(vkCode, 0, KEYEVENTF_KEYDOWN, 0)
# time.sleep(1)
# ctypes.windll.user32.keybd_event(vkCode, 0, KEYEVENTF_KEYDOWN, 0)
# time.sleep(1)
# """ver 1"""
# # ctypes.windll.user32.keybd_event(0xdb, 0, KEYEVENTF_KEYDOWN, 0)
# """ver 2"""
# ctypes.windll.user32.keybd_event(219, 0, KEYEVENTF_KEYDOWN, 0)


""" nie dziala """
# ck._prepare()
# pyautogui.press('-')


""" moglby dzialac """
# ck._prepare()
# pyautogui.press('ß')



