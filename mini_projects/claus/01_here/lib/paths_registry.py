import pathlib as pl
import os

class PathsRegistry:
    records = pl.Path(os.environ["CLAUS"]).joinpath("db").joinpath("niem_60").joinpath("wav")
    temp = pl.Path(__file__).parent.parent.joinpath('temp')
