import pathlib as pl

""" maksymalna liczba fullwordow z jednego idioma """
FREQUENCY_LIM = 10


with open(pl.Path(__file__).parent.joinpath('static', 'klaus_dictionary.txt'), 'r', encoding='utf8') as infile:
    db_raw = infile.readlines()

def get_fullwords(db_raw):
    fullwords = []
    for line in db_raw:
        fullwords.append(line.split('=')[0][:-1])
    return fullwords


def get_idioms_to_fullwords(fullwords):
    idioms_to_fullwords = {}
    for fullword in fullwords:
        idioms = fullword.lower().split(' ')
        for idiom in idioms:
            idioms_to_fullwords.setdefault(idiom, []).append(fullword)
    return idioms_to_fullwords


def filter_most_common(idioms_to_fullwords):
    global FREQUENCY_LIM
    most_common = {}
    less_common = {}
    for k, v in idioms_to_fullwords.items():
        if len(v) >= FREQUENCY_LIM:
            most_common[k] = v
        else:
            less_common[k] = v
    return less_common, most_common

fullwords = get_fullwords(db_raw)
idioms_to_fullwords = get_idioms_to_fullwords(fullwords)
idioms_to_fullwords, most_common = filter_most_common(idioms_to_fullwords)







# frequent = {}
# for k, v in idioms_to_fullwords.items():
#     if len(v) >= 10:
#         frequent[k] = v


# from collections import Counter
# c1 = Counter([len(elem) for elem in idioms_to_fullwords.values()])
# print(c1.most_common(20))


# for line in db_raw:
#     if line.split('=')[0][-1] != 'a':
#         print('err')