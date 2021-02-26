import ctypes
import pyautogui


def get_klid():
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    curr_window = user32.GetForegroundWindow()
    thread_id = user32.GetWindowThreadProcessId(curr_window, 0)
    klid = user32.GetKeyboardLayout(thread_id)
    klid_prim = str(hex(klid & (2 ** 16 - 1)))[2:].zfill(8)
    return klid_prim


class KeyboardLayout:
    polish_klid = '00000409'
    german_klid = '00000407'

    @staticmethod
    def change_keyboard_layout(klid):
        while get_klid() != klid:
            pyautogui.hotkey('winleft', 'space')

    @classmethod
    def to_polish(cls):
        KeyboardLayout.change_keyboard_layout(cls.polish_klid)

    @classmethod
    def to_german(cls):
        KeyboardLayout.change_keyboard_layout(cls.german_klid)


# a1 = '00000409' # to switch to english (pl or us)
# a2 = '00000407' # to switch to german

# pyautogui.hotkey('winleft', 'space')
# klid = get_klid()

# KeyboardLayout.change_keyboard_layout('00000409') #PL
# KeyboardLayout.change_keyboard_layout('00000407') #DE




