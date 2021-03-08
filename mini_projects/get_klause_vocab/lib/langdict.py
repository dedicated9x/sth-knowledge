from mini_projects.get_klause_vocab.config import FREQUENCY_LIM
import pandas as pd

class LangDict:
    def __init__(self, path_to_excel):
        self.df_ = pd.read_excel(path_to_excel, engine='openpyxl')
        self.fullwords = self.df_['DE'].astype('str')
        self.idioms_to_fullwords = self._get_idioms_to_fullwords()


    def has_idiom(self, idiom):
        return idiom in self.idioms_to_fullwords

    def _get_idioms_to_fullwords(self, filtered=True):
        idioms_to_fullwords = {}
        for fullword in self.fullwords:
            idioms = fullword.lower().split(' ')
            for idiom in idioms:
                idioms_to_fullwords.setdefault(idiom, []).append(fullword)
        if filtered == True:
            idioms_to_fullwords, most_common = self._filter_most_common(idioms_to_fullwords)
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



from mini_projects.get_klause_vocab.klaus.paths_registry import PathsRegistry
path_ = PathsRegistry.basic_db_txt.with_name('dict_klaus.xlsx')
ld = LangDict(path_)

assert ld.has_idiom('abbruch')
assert not ld.has_idiom('abbruch2')


"""correctness test"""
# from mini_projects.get_klause_vocab.klaus.paths_registry import PathsRegistry
# path_ = PathsRegistry.basic_db_txt.with_name('dict_klaus.xlsx')
# ld = LangDict(path_)
# idioms_to_fullwords = ld.idioms_to_fullwords
#
# import itertools
# fullwords = list(itertools.chain(*[idioms_to_fullwords[k] for k in idioms_to_fullwords.keys()]))
# assert len(idioms_to_fullwords) == 9318
# assert len(fullwords) == 10714


# TODO zaladowac fullwords