import pathlib as pl
from pydub import AudioSegment

root = pl.Path(rf"C:\Users\devoted\Desktop\afirmacja")

sound1 = AudioSegment.from_file(root.joinpath('ambient.wav'))
sound2 = AudioSegment.from_file(root.joinpath('speech.wav'))
combined = sound1.overlay(sound2)
combined.export(root.joinpath('combined.wav'), format='wav')


