import pathlib as pl
import pandas as pd
from bs4 import BeautifulSoup

path_txt = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\slownik.txt")
with open(path_txt, 'r', encoding='utf8') as infile:
    lines = infile.readlines()

def get_wt_number(single_row_df: pd.DataFrame):
    wp_tag = BeautifulSoup(single_row_df['tag'], 'lxml')
    return len(wp_tag.find_all('w:t'))

def get_text(single_row_df: pd.DataFrame):
    wp_tag = BeautifulSoup(single_row_df['tag'], 'lxml')
    return ''.join([elem.contents[0] for elem in wp_tag.find_all('w:t')])

def get_dest(single_row_df: pd.DataFrame):
    wt_number = single_row_df['wt_number']
    if wt_number in {0, 5}:
        return 'excluded'
    elif wt_number in {1, 2}:
        return 'additional'
    elif wt_number in {3, 4} or wt_number > 14:
        return 'hopeless'
    else:
        return 'core'

def get_no_dash(single_row_df: pd.DataFrame):
    tag = BeautifulSoup(single_row_df['tag'], 'lxml')
    elems = [elem.contents[0] for elem in tag.find_all('w:t')]
    return elems.count('–')

def get_no_pipes(single_row_df: pd.DataFrame):
    text = single_row_df['text']
    return text.count('|')

def get_no_braces(single_row_df: pd.DataFrame):
    text = single_row_df['text']
    return f'{text.count("[")}{text.count("]")}'

def get_clean_part(df_):
    df_ = df_.assign(wt_number=df_.apply(get_wt_number, axis=1))
    df_ = df_.assign(text=df_.apply(get_text, axis=1))
    df_ = df_.assign(dest=df_.apply(get_dest, axis=1))

    df_core = df_[df_['dest'] == 'core']
    df_core = df_core.assign(no_dash=df_core.apply(get_no_dash, axis=1))
    df_core = df_core[df_core['no_dash'] == 1]                                  # 0.5% (faile)
    df_core = df_core.assign(no_pipes=df_core.apply(get_no_pipes, axis=1))
    df_core = df_core[df_core['no_pipes'] == 2]                                 #1.5% (faile), 0.01% (brak [])
    df_core = df_core.assign(no_braces=df_core.apply(get_no_braces, axis=1))
    df_core = df_core[(df_core['no_braces'] == '11') | (df_core['no_braces'] == '00')]
    df_core = df_core[['text', 'no_braces', 'tag']]

    return df_core


df = pd.DataFrame().assign(tag=pd.Series(lines))
part_pn = df.iloc[137:17443]
part_np = df.iloc[17485:35481]

pn_core = get_clean_part(part_pn)
np_core = get_clean_part(part_np)


pn_core['np'] = np_core.apply(lambda x: False, axis=1)
np_core['np'] = np_core.apply(lambda x: True, axis=1)


data2 = [
    ('pn', pn_core[pn_core['no_braces'] == '11'].iloc[0]),
    ('pn', pn_core[pn_core['no_braces'] == '00'].iloc[0]),
    ('np', np_core[np_core['no_braces'] == '11'].iloc[0]),
    ('np', np_core[np_core['no_braces'] == '00'].iloc[0])
]


def get_components(single_row_df: pd.DataFrame):
    row = single_row_df
    wts = [elem.contents[0] for elem in BeautifulSoup(row['tag'], 'lxml').find_all('w:t')]
    dash_idx = wts.index('–')
    first, second_and_desc = ''.join(wts[:dash_idx]), ''.join(wts[(dash_idx + 1):])
    second, desc = second_and_desc.split('|')[:2]

    if row['no_braces'] == '11':
        first = first.split('[')[0]

    first, second, desc = first.strip(), second.strip(), desc.strip()

    if row['np'] == True:
        single_row_df['de'] = first
        single_row_df['pl'] = second
    else:
        single_row_df['de'] = second
        single_row_df['pl'] = first
    single_row_df['desc'] = desc
    return single_row_df

pn_core = pn_core.apply(get_components, axis=1)
np_core = np_core.apply(get_components, axis=1)

pn_core = pn_core[['de', 'pl', 'desc']]
np_core = np_core[['de', 'pl', 'desc']]

result = pn_core
result.to_excel(path_txt.with_suffix('.xlsx'))


"""0.78% -> 0.86%"""
# feature = 'desc'
# s1 = set(pn_core[feature].to_list())
# s2 = set(np_core[feature].to_list())
# len(s1 & s2)