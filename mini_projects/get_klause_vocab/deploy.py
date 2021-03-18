import pathlib as pl
import pandas as pd
from mini_projects.claus.lib.klaus_dir import KlausDir

def get_words_done_all(path_sbusr_):
    with open(path_sbusr_, 'r') as infile:
        lines = infile.readlines()
    lines = list(filter(lambda x: len(x)> 3, lines))
    lines = lines[1:]
    lines = [''.join(l.split('\x00')) for l in lines]
    words_done_all = pd.DataFrame().assign(DE=pd.Series(lines).apply(lambda x: x[12:].split('=')[0].strip()))
    return words_done_all, int(lines[-1].split(' ')[1])

def prepare_words_new(words_new_):
    max_n = KlausDir(path_klaus)._get_max_record_number()
    words_new_ = words_new_.assign(FILENAME=words_new_['ID'].apply(lambda x: f"{x - 10000 + max_n + 1}.wav"))
    words_new_['KLAUSID'] = words_new_.reset_index(drop=True).index.to_series().apply(lambda x: f"{2 * (x + 1) + max_klausid}")
    def calculate_entry(row):
        row['ENTRY'] = f"99 {row['KLAUSID']} 24 {row['DE']} = {row['PL']} == =  ={row['FILENAME']} = == n = = ="
        return row
    words_new_ = words_new_.apply(calculate_entry, axis=1)
    return words_new_




path_klaus = pl.Path(rf'C:\Users\devoted\Desktop\ksiazki_nat_fp\current_klaus\Profesor Klaus 6.0 S³ownictwo')
path_sound_db = pl.Path(rf'C:\Users\devoted\Desktop\ksiazki_nat_fp\sound_db')

path_excel = pl.Path(__file__).parent.joinpath('static', 'fp_final.xlsx')
path_sbusr = path_klaus.joinpath('db', 'niem_60', 'sbusr_u.txt')

words_dict = pd.read_excel(path_excel, engine='openpyxl')
words_done_all, max_klausid = get_words_done_all(path_sbusr)
words_done_dict = pd.merge(words_dict, words_done_all, how='inner', on='DE')
words_new = pd.concat([words_dict, words_done_dict]).drop_duplicates(keep=False)
words_new = prepare_words_new(words_new)

# TODO reset_index sie nie skasowal. Trzeba to zrobic range(n)