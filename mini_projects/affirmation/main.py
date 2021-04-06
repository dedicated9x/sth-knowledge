from mini_projects.claus.lib.text_to_mp3_converter import TextToMp3Converter
from mini_projects.claus.lib.wav_player import WavPlayer
from mini_projects.claus.lib.wav_creator import WavCreator
from mini_projects.affirmation.ssml import input2ssml
import pathlib as pl

path_root = pl.Path(rf"C:\Users\devoted\Desktop\afirmacja")

input_ = [
    ('<prosody rate="90%">Analiza rynku atakuje znienacka</prosody>. Niewprawionego uczonego sprowadzi ona na bagna. A Ty? Czy <emphasis level="strong">masz</emphasis> na nią jakiś sposób?', 6),
    ('Pomysły pomysły pomysły. Jak oryginalny pomysł by nie był i tak w temacie działa już kilka firm. Czy <emphasis level="strong">zraża</emphasis> Cię to?', 10),
    ('Życie to gra. <emphasis level="strong">Kiedy rozpoczęła się</emphasis> <emphasis level="strong">Twoja?</emphasis>', 8),
    ('Kto często zmienia kierunek ten stoi w miejscu. A jaki jest Twój kierunek?', 6),
    ('Produkty produkty produkty. Bywają złożone. Potrzeba wielu różnych rąk je by je stworzyć. Czy aby na pewno?', 12),
    ('<prosody rate="80%">Niejednoznaczność</prosody>. Człek przeciętny unika jej jak ognia. Jednak nie Ty Pawle. W końcu <emphasis level="moderate">jesteś</emphasis>', 1),
]

text = input2ssml(input_)

"""TESTING"""
# text = """<speak>
# <amazon:effect name="whispered">
#     <prosody rate="90%">Analiza dupy atakuje znienacka</prosody>. Niewprawionego uczonego sprowadzi ona na bagna. A Ty? Czy <emphasis level="strong">masz</emphasis> na nią jakiś sposób?
# </amazon:effect>
# </speak>"""


sound = TextToMp3Converter().convert(text, voice_id='Maja', is_simple=False)
path_to_wav = WavCreator(workdir=path_root.joinpath("temp")).create_wav(sound, 'speech.mp3')
WavPlayer.play(path_to_wav.__str__())
