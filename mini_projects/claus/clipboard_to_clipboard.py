from lib.clipboard_controller import ClipboardController
from lib.text_to_wav_converter import KlausTextToWavConverter
from mini_projects.claus.lib.paths_registry import PathsRegistry


if __name__ == "__main__":
    text = ClipboardController.get_clipboard_value()
    path_to_wav = KlausTextToWavConverter(PathsRegistry.klaus).convert(text, verbose=1)
    ClipboardController.save_to_clipboard(str(path_to_wav))

"""Pierwsze slowko. Nie usuwac."""
