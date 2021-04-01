from lib.clipboard_controller import ClipboardController
from lib.text_to_wav_converter import KlausTextToWavConverter
from mini_projects.claus.lib.paths_registry import PathsRegistry

# text = '''<speak>Mary had a little lamb <break time="3s"/>Whose fleece was white as snow.</speak>'''



if __name__ == "__main__":
    text = ClipboardController.get_clipboard_value()
    path_to_wav = KlausTextToWavConverter(PathsRegistry.klaus).convert(text, verbose=1)
    ClipboardController.save_to_clipboard(str(path_to_wav))


# TODO wyekstrahoanie metody na to
# Czy risercz trzeba robić na sto procent? A kiedy należałoby go zaprzestać?
# Polskie slowko
# Niemieckie slownko