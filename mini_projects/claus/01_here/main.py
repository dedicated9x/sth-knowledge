from pathlib import Path
import os
import shutil

from lib.voice_synthetizer import VoiceSynthetizer
from lib.clipboard_controller import ClipboardController
from lib.wav_player import WavPlayer

# TODO w sumie auth powinien isc do enva, a link do klausa moze byc w configu
PATH_TO_AUTH = rf"C:\Users\devoted\Documents\nataly_aws_auth.csv"
PROSODY_RATE = 70

class WavCreator:
    # TODO tidy do synthetizera
    cwd = None

    @classmethod
    def tidy(cls):
        cls.cwd = Path(os.path.realpath(__file__)).parent
        path_to_temp = cls.cwd.joinpath("temp")
        shutil.rmtree(path_to_temp)
        path_to_temp.mkdir()
        # return cwd

    # @staticmethod
    # def convert_mp3_to_wav(src, dst, cwd):

    @classmethod
    def convert_mp3_to_wav(cls, src, dst):
        path_to_fffmpeg = cls.cwd.joinpath("ffmpeg").joinpath("bin").joinpath("ffmpeg.exe")
        os.system(f"{path_to_fffmpeg} -i {src} -acodec pcm_s16le -ac 1 -ar 16000 {dst}")
        return dst



if __name__ == "__main__":
    WavCreator.tidy()
    path_to_mp3, sound = VoiceSynthetizer().make_sound_from_text(PATH_TO_AUTH, PROSODY_RATE)
    path_to_wav = WavCreator.convert_mp3_to_wav(path_to_mp3, path_to_mp3.with_suffix(".wav"))
    WavPlayer.play(str(path_to_wav))
    ClipboardController.save_to_clipboard(str(path_to_wav))

# TODO WavCreator, ktory ma opcje inform=True

# TODO wyekstrahuj aws z dwoma interfejsami (z clipa i z listy)

"""nier automata"""
"""dzieci"""
"""najnowsze"""




