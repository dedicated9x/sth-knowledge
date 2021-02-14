import os
import shutil
from pathlib import Path
import pathlib as pl

from lib.voice_synthetizer import VoiceSynthetizer
from lib.clipboard_controller import ClipboardController
from lib.wav_player import WavPlayer
from lib.klaus_state_analyzer import KlausStateAnalyzer
from lib.paths_registry import PathsRegistry


# TODO w sumie auth powinien isc do enva, a link do klausa moze byc w configu
PATH_TO_AUTH = rf"C:\Users\devoted\Documents\nataly_aws_auth.csv"
PROSODY_RATE = 70

class WavCreator:
    # cwd = None

    @staticmethod
    def get_path_to_output():
        max_record_number = KlausStateAnalyzer().get_max_record_number()
        new_record_filename = f"{max_record_number + 1}.mp3"
        new_record_path = PathsRegistry.temp.joinpath(new_record_filename)
        return new_record_path


    @classmethod
    def _tidy(cls):
        cls.cwd = Path(os.path.realpath(__file__)).parent
        path_to_temp = cls.cwd.joinpath("temp")
        shutil.rmtree(path_to_temp)
        path_to_temp.mkdir()

    @classmethod
    def _save_mp3(cls, output_path, sound):
        file = open(output_path, 'wb')
        file.write(sound)
        file.close()

    @classmethod
    def convert_mp3_to_wav(cls, src, dst):
        path_to_fffmpeg = cls.cwd.joinpath("ffmpeg").joinpath("bin").joinpath("ffmpeg.exe")
        os.system(f"{path_to_fffmpeg} -i {src} -acodec pcm_s16le -ac 1 -ar 16000 {dst}")

    @classmethod
    def create_wav(cls, sound):
        path_to_mp3 = WavCreator.get_path_to_output()
        path_to_wav = path_to_mp3.with_suffix(".wav")

        cls._tidy()
        cls._save_mp3(path_to_mp3, sound)
        cls.convert_mp3_to_wav(path_to_mp3, path_to_wav)
        return path_to_wav



if __name__ == "__main__":
    sound = VoiceSynthetizer().make_sound_from_text(PATH_TO_AUTH, PROSODY_RATE)
    path_to_wav = WavCreator.create_wav(sound)

    WavPlayer.play(str(path_to_wav))
    ClipboardController.save_to_clipboard(str(path_to_wav))

# TODO 1 metoda tworzaca sciezke ma byc wrzucona do creatora

# TODO wyekstrahuj aws z dwoma interfejsami (z clipa i z listy)

"""roszadza"""
"""przenoszenie"""
"""reefaktore"""




