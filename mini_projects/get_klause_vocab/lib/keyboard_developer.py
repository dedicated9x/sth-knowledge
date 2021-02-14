
def add_german_vk_codes():
    import pyautogui._pyautogui_win as pyautogui_win
    pyautogui_win.keyboardMapping.update({
        'ß': 0xDB,
        'ü': 0xBA,
        'ö': 0xC0,
        'ä': 0xDE,
        'Ü': 0xBA + 0x100,
        'Ö': 0xC0 + 0x100,
        'Ä': 0xDE + 0x100
    })
