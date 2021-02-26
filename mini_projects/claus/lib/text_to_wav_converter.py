from mini_projects.claus.lib.text_to_mp3_converter import TextToMp3Converter
from mini_projects.claus.lib.wav_player import WavPlayer
from mini_projects.claus.lib.wav_creator import WavCreator
from mini_projects.claus.lib.paths_registry import PathsRegistry
from mini_projects.claus.lib.klaus_dir import KlausDir


class KlausTextToWavConverter:
    def __init__(self, path_to_klaus_dir):
        self.path_to_klaus_dir_= path_to_klaus_dir

    def convert(self, text, verbose=0):
        sound = TextToMp3Converter().convert(text)
        filename = KlausDir(self.path_to_klaus_dir_).get_next_available_record_filename()
        path_to_wav = WavCreator(workdir=PathsRegistry.temp).create_wav(sound, filename)
        if verbose == 1:
            WavPlayer.play(str(path_to_wav))
        return path_to_wav
