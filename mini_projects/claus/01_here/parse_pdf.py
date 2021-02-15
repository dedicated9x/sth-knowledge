import pathlib as pl

from tika import parser # pip install tika


path_pdf = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\Slownik-medyczny-polsko-niemiecki-niemiecko-polski-z-definicjami-hasel.pdf")
# with open(path_txt, 'r', encoding='utf8') as infile:
#     lines = infile.readlines()


raw = parser.from_file(str(path_pdf))
print(raw['content'])