import pathlib as pl
import os

class PathsRegistry:
    klaus = pl.Path(os.environ["CLAUS"])
    # ffmpeg = pl.Path(__file__).parent.parent.joinpath("ffmpeg").joinpath("bin").joinpath("ffmpeg.exe")
    ffmpeg = pl.Path(rf"C:\Users\devoted\Documents\bin\ffmpeg\bin\ffmpeg.exe")
    temp = pl.Path(__file__).parent.parent.joinpath('temp')

