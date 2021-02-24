import pathlib as pl

class PathsRegistry:
    basic_db_txt = pl.Path(__file__).parent.parent.joinpath('static', 'klaus_dictionary.txt')
