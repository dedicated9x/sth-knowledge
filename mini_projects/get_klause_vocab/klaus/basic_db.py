from mini_projects.get_klause_vocab.klaus.paths_registry import PathsRegistry
from mini_projects.get_klause_vocab.config import FREQUENCY_LIM

class BasicDB:
    @staticmethod
    def get_lines():
        with open(PathsRegistry.basic_db_txt, 'r', encoding='utf8') as infile:
            lines = infile.readlines()
        return lines

    @classmethod
    def get_fullwords(cls):
        lines = cls.get_lines()
        fullwords = []
        for line in lines:
            fullwords.append(line.split('=')[0][:-1])
        return fullwords

    @classmethod
    def get_idioms_to_fullwords(cls, filtered=True):
        fullwords = cls.get_fullwords()
        idioms_to_fullwords = {}
        for fullword in fullwords:
            idioms = fullword.lower().split(' ')
            for idiom in idioms:
                idioms_to_fullwords.setdefault(idiom, []).append(fullword)
        if filtered == True:
            idioms_to_fullwords, most_common = cls._filter_most_common(idioms_to_fullwords)
        return idioms_to_fullwords

    @staticmethod
    def _filter_most_common(idioms_to_fullwords):
        most_common = {}
        less_common = {}
        for k, v in idioms_to_fullwords.items():
            if len(v) >= FREQUENCY_LIM:
                most_common[k] = v
            else:
                less_common[k] = v
        return less_common, most_common


# TODO 1 odpal ten stary test (i podmien na parsowanie csv)
"""klaus = convert .txt to clean .xlsx"""
# import pandas as pd
# import pathlib as pl
#
# def func1(row):
#     line = row['LINES']
#     row['DE'], row['PL'] = [elem.strip() for elem in line.rstrip('\n').split('=')]
#     return row
#
# lines = BasicDB().get_lines()
# lines = pd.Series(lines)
# lines = pd.DataFrame().assign(LINES=lines)
# lines2 = lines.apply(func1, axis=1)
# df = lines2[['DE', 'PL']]
# path_output = pl.Path(__file__).parent.parent.joinpath('static', 'dict_klaus.xlsx')
# df.to_excel(path_output, index=False)
# df1 = pd.read_excel(path_output, engine='openpyxl')

"""med = convert .xlsx to clean .xlsx"""
# import pandas as pd
# import pathlib as pl
# path_ = pl.Path(__file__).parent.parent.joinpath('static', 'dtcpro_1_v3.xlsx')
# df1 = pd.read_excel(path_, engine='openpyxl')
# df2 = df1[['corr_de', 'pl']]
# df3 = df2.rename(columns={'corr_de': 'DE', 'pl': 'PL'})
# path_output = pl.Path(__file__).parent.parent.joinpath('static', 'dict_med.xlsx')
# df3.to_excel(path_output, index=False)