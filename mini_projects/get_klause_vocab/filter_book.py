import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt

path_excel = pl.Path(__file__).parent.joinpath('static', 'book_ocred.xlsx')
df = pd.read_excel(path_excel, engine='openpyxl')




"""established short limit"""
# counts = df.apply(lambda x: len(x['TEXT']), axis=1).tolist()
# plt.hist(counts, bins=20)


"""analyze short texts"""
# short = df[df['TEXT'].str.len() < 700]
