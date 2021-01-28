import pyautogui
import time
"""
pyautogui.position() # Get the XY position of the mouse.
"""

"""
1000 -  500s    - 8 minut
10000 - 5000s   - 80 minut :)
"""


for i in range(305):
    time.sleep(0.5)
    pyautogui.click(x=1382, y=705)
    pyautogui.press('down')

