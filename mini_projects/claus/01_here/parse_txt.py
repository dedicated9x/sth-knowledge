import pathlib as pl
import pandas as pd
from bs4 import BeautifulSoup

path_txt = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\slownik.txt")
with open(path_txt, 'r', encoding='utf8') as infile:
    lines = infile.readlines()



# def wt_number(wp_tag):
#     return len(wp_tag.find_all('w:t'))
#
# def get_text(wp_tag):
#     return ''.join([elem.contents[0] for elem in wp_tag.find_all('w:t')])


def wt_number_pd(single_row_df: pd.DataFrame):
    wp_tag = BeautifulSoup(single_row_df['tag'], 'lxml')
    return len(wp_tag.find_all('w:t'))

def get_text_pd(single_row_df: pd.DataFrame):
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

def get_dash(single_row_df: pd.DataFrame):
    tag = BeautifulSoup(single_row_df['tag'], 'lxml')
    elems = [elem.contents[0] for elem in tag.find_all('w:t')]
    return '–' in elems

def get_vert(single_row_df: pd.DataFrame):
    text = single_row_df['text']
    return text.count('|')

def get_braces_valid(single_row_df: pd.DataFrame):
    text = single_row_df['text']
    return f'{text.count("[")}{text.count("]")}' in {'00', '11'}


def get_clean_part(df_):
    df_ = df_.assign(wt_number=df_.apply(wt_number_pd, axis=1))
    df_ = df_.assign(text=df_.apply(get_text_pd, axis=1))
    df_ = df_.assign(dest=df_.apply(get_dest, axis=1))

    df_core = df_[df_['dest'] == 'core']
    df_core = df_core[['text', 'wt_number', 'dest', 'tag']]
    df_core = df_core.assign(dash=df_core.apply(get_dash, axis=1))
    df_core = df_core[df_core['dash'] == True]
    df_core = df_core.assign(vert=df_core.apply(get_vert, axis=1))
    df_core = df_core[df_core['vert'] == 2]
    df_core = df_core.assign(braces_valid=df_core.apply(get_braces_valid, axis=1))
    df_core = df_core[df_core['braces_valid'] == True]
    return df_core


df = pd.DataFrame().assign(tag=pd.Series(lines))
part_np = df.iloc[137:17443]
part_pn = df.iloc[17485:35481]


np_core = get_clean_part(part_np)
pn_core = get_clean_part(part_pn)

# TODO hardkorowe wyprowadzenie wszystkich czesci

target = pn_core








def get_part_desc(single_row_df: pd.DataFrame):
    text = single_row_df['text']
    return text.split('|')[1]

# text = pn_core.iloc[1]['text']
# part_desc = text.split('|')[1]


np_core = np_core.assign(part_desc=np_core.apply(get_part_desc, axis=1))
pn_core = pn_core.assign(part_desc=pn_core.apply(get_part_desc, axis=1))

# s1 = set(np_core['part_desc'].to_list())
# s2 = set(pn_core['part_desc'].to_list())


"""0.78% sparsowalo sie idealnie"""
# not_two = pn_core[pn_core['vert'] != 2]
# not_two = not_two[['text', 'vert', 'wt_number']]


"""szukanie dasha"""
# tag = BeautifulSoup(pn_core.iloc[0]['tag'], 'lxml')
# elems = [elem.contents[0] for elem in tag.find_all('w:t')]
# '–' in elems
# '-' in elems

"""liczebnosc"""
# df['wt_number'].value_counts().sort_index()

"""casy - analiza, co w nich jest"""
# case = 10
# z1 = df[df['wt_number'] == case]['text_text']