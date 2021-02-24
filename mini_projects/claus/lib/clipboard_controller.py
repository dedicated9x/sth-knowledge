import pyperclip
from tkinter import Tk


class ClipboardController:
    @staticmethod
    def save_to_clipboard(string_):
        pyperclip.copy(string_)
        pyperclip.paste()

    @staticmethod
    def get_clipboard_value():
        return Tk().clipboard_get()
