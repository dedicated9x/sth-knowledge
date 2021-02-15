import pathlib as pl
import pandas as pd
from bs4 import BeautifulSoup

path_txt = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\Slownik-medyczny-polsko-niemiecki-niemiecko-polski-z-definicjami-hasel.txt")
with open(path_txt, 'r', encoding='utf8') as infile:
    lines = infile.readlines()


def wt_number(wp_tag):
    return len(wp_tag.find_all('w:t'))

def get_text(wp_tag):
    return ''.join([elem.contents[0] for elem in wp_tag.find_all('w:t')])


def wt_number_pd(single_row_df: pd.DataFrame):
    wp_tag = BeautifulSoup(single_row_df['text'], 'lxml')
    return wt_number(wp_tag)

def get_text_pd(single_row_df: pd.DataFrame):
    wp_tag = BeautifulSoup(single_row_df['text'], 'lxml')
    return get_text(wp_tag)


df = pd.DataFrame().assign(text=pd.Series(lines))
df = df.assign(wt_number=df.apply(wt_number_pd, axis=1))
df = df.assign(text_text=df.apply(get_text_pd, axis=1))

"""liczebnosc"""
# df['wt_number'].value_counts().sort_index()

"""casy - analiza, co w nich jest"""
case = 10
z1 = df[df['wt_number'] == case]['text_text']

"""
0,5     -> do wyjebnia 
1,2     -> mamy te "SZCZEGOLNE PRZYPADKI" (uff...)
3,4     -> ulepy (poprawka)
6:14    -> MIESKO
"""
