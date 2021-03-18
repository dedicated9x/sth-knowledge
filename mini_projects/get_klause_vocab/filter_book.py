import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt
from mini_projects.get_klause_vocab.klaus.paths_registry import PathsRegistry
from mini_projects.get_klause_vocab.lib.langdict import LangDict


def text2idioms(row):
    text = row['TEXT']
    idioms = list(set(text.split(' ')))
    res_df = pd.DataFrame().assign(IDIOM=idioms)
    res_df['CHAPTER'] = row['CHAPTER']
    return res_df


path_excel = pl.Path(__file__).parent.joinpath('static', 'book_ocred.xlsx')
df = pd.read_excel(path_excel, engine='openpyxl')

df['TEXT'] = df.apply(lambda x: ' '.join(x['TEXT'].split('\n')), axis=1)  # newlines
df['TEXT'] = df.apply(lambda x: ''.join([c if c.isalpha() else " " for c in x['TEXT']]), axis=1)  # non-alpha chars
df['TEXT'] = df['TEXT'].str.lower()

dict_klaus = LangDict(PathsRegistry.basic_db_txt.with_name('dict_klaus.xlsx'))
dict_med = LangDict(PathsRegistry.basic_db_txt.with_name('dict_med.xlsx'))

idioms = df.assign(TEXT=df['TEXT'].str.split(' ')).explode('TEXT')[['TEXT', 'CHAPTER']].rename(columns={'TEXT': 'IDIOM'})
idioms = idioms.groupby('IDIOM')['CHAPTER'].min().to_frame().reset_index().sort_values(by='CHAPTER')  # agg (1st occurence instead of all)
idioms = idioms[idioms['IDIOM'].str.len() > 3]  #exclude short words
idioms['CLASS'] = idioms['IDIOM'].apply(lambda x: ''.join([str(int(bool_)) for bool_ in [dict_med.has_idiom(x), dict_klaus.has_idiom(x)]]))
idioms['IDIOM'].apply(lambda x: ''.join([str(int(bool_)) for bool_ in [dict_med.has_idiom(x), dict_klaus.has_idiom(x)]]))
idioms = idioms[idioms['CLASS'].map({'10': 1, '11': 1, '01': 0, '00': 0}) == 1]  #filter out nonvaluable idioms
idioms['WORD'] = idioms['IDIOM'].map(dict_med.idioms_to_fullwords)

fullwords = idioms.explode('WORD')[['WORD', 'CHAPTER']]
fullwords = fullwords.groupby('WORD')['CHAPTER'].min().to_frame().reset_index().sort_values(by='CHAPTER') #drop duplicates (min() is important)

result = pd.merge(fullwords.rename(columns={'WORD': 'DE'}), dict_med.df_, on='DE')[['DE', 'PL', 'CHAPTER']]

def get_count(group):
    group['COUNT'] = group['DE'].count()
    return group

to_export = result.groupby('DE').apply(get_count).sort_values(by=['COUNT', 'DE'])
to_export.to_excel(path_excel.with_name('fp.xlsx'), index=False)




# z1 = result.groupby('DE')['DE'].count()
# z2 = pd.DataFrame().assign(DE=z1.index, COUNT=z1.values)


"""dziwne literki"""
# all_letters = 'ÃÄÖÜßàáâäåçèéêëíñóôöøüıšμﬁﬂ'
# letters = 'Ãàáâåçèéêíñóôøıšμﬁﬂ'
# bool(set('REPLACE THAT!!!') & set(letters))
# bad_names = idioms[idioms.apply(lambda row: bool(set(row['IDIOM']) & set(letters)), axis=1)]

"""proba na series"""
# def text2idioms2(row):
#     text = row['TEXT']
#     idioms = list(set(text.split(' ')))
#     idioms = pd.Series(idioms)
#     return idioms
"""established short limit"""
# counts = df.apply(lambda x: len(x['TEXT']), axis=1).tolist()
# plt.hist(counts, bins=20)
"""analyze short texts"""
# short = df[df['TEXT'].str.len() < 700]
