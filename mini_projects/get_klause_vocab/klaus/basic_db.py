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
