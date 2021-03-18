# import pathlib as pl
# import pandas as pd
# from mini_projects.get_klause_vocab.config import PATH_TO_KLAUS_DIR
# from mini_projects.claus.lib.text_to_wav_converter import KlausTextToWavConverter
# import shutil
#
# path_excel = pl.Path(__file__).parent.joinpath('static', 'fp_final.xlsx')
# path_output = pl.Path(rf'C:\Users\devoted\Desktop\ksiazki_nat_fp\sound_db')
#
# df = pd.read_excel(path_excel, engine='openpyxl')
#
#
# for e in df.iterrows():
#     row = e[1]
#     word_de = row['DE']
#     path_to_wav = KlausTextToWavConverter(pl.Path(PATH_TO_KLAUS_DIR)).convert(word_de.rstrip('-'), verbose=0)
#     shutil.copyfile(path_to_wav, path_output.joinpath(f"{row['ID']}.wav"))
#     print(f"{row['ID']}.wav")

