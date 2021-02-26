import pathlib as pl
import pandas as pd


TYPE_TO_ARTICLE = {
    'nm': 'der',
    'nf': 'die',
    'nnt': 'das'
}
def get_corrected_de(row: pd.DataFrame):
    global TYPE_TO_ARTICLE
    try:
        word_de = row['de']
        subwords = word_de.split(',')
        corrected_subwords = []
        for subword in subwords:
            part1, part2 = subword.split('(')
            core = part1.strip()
            type_ = part2[:-1]
            article = TYPE_TO_ARTICLE.get(type_)
            corrected_subword = core
            if article is not None:
                corrected_subword = ' '.join([article, corrected_subword])
            corrected_subwords.append(corrected_subword)
        corrected_word = ', '.join(corrected_subwords)
        row['corr_de'] = corrected_word
        row['syn'] = len(subwords)
        row['is_reg'] = 1
    except:
        row['corr_de'] = word_de
        row['is_reg'] = 0
    return row


def get_add_suffix_to_pl(row: pd.DataFrame):
    if str(row['syn']) != '1':
        row['pl'] = f"{str(row['pl'])} ({str(row['syn'])})"
    return row


"""v1 -> v2"""
# path_dtcpro = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\dtcpro_1.xlsx")
# df = pd.read_excel(path_dtcpro, engine='openpyxl')
# df = df.apply(get_corrected_de, axis=1)
# df = df.sort_values(by='is_reg')
# df = df[['part1', 'syn', 'corr_de', 'pl', 'is_reg']]
# df.to_excel(path_dtcpro.with_name('dtcpro_1_v2.xlsx'))


"""v1 -> v3"""
# path_dtcpro = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\dtcpro_1_v2.xlsx")
# df = pd.read_excel(path_dtcpro, engine='openpyxl')
# df = df[df['corr_de'].isna() == False]
# df = df.apply(get_add_suffix_to_pl, axis=1)
# df = df[['part1', 'corr_de', 'pl']]
# df.to_excel(path_dtcpro.with_name('dtcpro_1_v3.xlsx'))

