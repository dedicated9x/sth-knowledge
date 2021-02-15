from lib.voice_synthetizer import VoiceSynthetizer
from lib.clipboard_controller import ClipboardController
from lib.wav_player import WavPlayer
from lib.wav_creator import WavCreator

# TODO w sumie auth powinien isc do enva, a link do klausa moze byc w configu
PATH_TO_AUTH = rf"C:\Users\devoted\Documents\nataly_aws_auth.csv"
PROSODY_RATE = 70

if __name__ == "__main__":
    # TODO auth i prosody do configa
    sound = VoiceSynthetizer().make_sound_from_text(PATH_TO_AUTH, PROSODY_RATE)
    path_to_wav = WavCreator.create_wav(sound)
    WavPlayer.play(str(path_to_wav))
    ClipboardController.save_to_clipboard(str(path_to_wav))


# TODO wyekstrahuj aws z dwoma interfejsami (z clipa i z listy)

"""roszadza"""
"""przenoszenie"""
"""reefaktore"""




