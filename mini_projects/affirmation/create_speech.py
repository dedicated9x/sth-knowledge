from mini_projects.claus.lib.text_to_mp3_converter import TextToMp3Converter
from mini_projects.claus.lib.wav_player import WavPlayer
from mini_projects.claus.lib.wav_creator import WavCreator
from mini_projects.claus.lib.paths_registry import PathsRegistry
from mini_projects.claus.lib.klaus_dir import KlausDir
from mini_projects.affirmation.ssml import input2ssml
import pathlib as pl
import os

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
text = """<speak>
<amazon:effect name="whispered">
    <prosody rate="90%">Analiza rynku atakuje znienacka</prosody>. Niewprawionego uczonego sprowadzi ona na bagna. A Ty? Czy <emphasis level="strong">masz</emphasis> na nią jakiś sposób?
</amazon:effect>
</speak>"""


# TODO wpisac to w jakis
sound = TextToMp3Converter().convert(text, voice_id='Maja', is_simple=False)

# TODO obyć się bez Klausa
filename = KlausDir(pl.Path(os.environ["CLAUS"])).get_next_available_record_filename()
path_to_wav = WavCreator(workdir=PathsRegistry.temp).create_wav(sound, filename)
WavPlayer.play(str(path_to_wav))
