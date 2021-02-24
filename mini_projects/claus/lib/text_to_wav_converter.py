from lib.text_to_mp3_converter import TextToMp3Converter
from lib.wav_player import WavPlayer
from lib.wav_creator import WavCreator
from lib.paths_registry import PathsRegistry
from lib.klaus_dir import KlausDir


class KlausTextToWavConverter:
    @staticmethod
    def convert(text, verbose=0):
        sound = TextToMp3Converter().convert(text)
        filename = KlausDir.get_next_available_record_filename()
        path_to_wav = WavCreator(workdir=PathsRegistry.temp).create_wav(sound, filename)
        if verbose == 1:
            WavPlayer.play(str(path_to_wav))
        return str(path_to_wav)
