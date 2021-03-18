import pathlib as pl
import pandas as pd

path_excel = pl.Path(__file__).parent.joinpath('static', 'fp_verified.xlsx')
df = pd.read_excel(path_excel, engine='openpyxl')
df_filtered = df[~(df['PL'].isna())]
df_final = df_filtered.sort_values(by='CHAPTER').drop(['COUNT'], axis=1).assign(ID=[10000 + i for i in range(df_filtered.shape[0])])

df_final.to_excel(path_excel.with_name('fp_final.xlsx'), index=False)