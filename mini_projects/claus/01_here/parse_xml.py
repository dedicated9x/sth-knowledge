from bs4 import BeautifulSoup
import pathlib as pl

path_xml = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\Slownik-medyczny-polsko-niemiecki-niemiecko-polski-z-definicjami-hasel.xml")
with open(path_xml, 'r', encoding='utf8') as infile:
    text = infile.read()

soup = BeautifulSoup(text, 'lxml')

z1 = soup.find('w:body')
z2 = soup.find_all('w:p')

z3 = z2[157].find_all('w:t')
for elem in z3:
    print(elem)

z4 = z2[159].find_all('w:t')
for elem in z4:
    print(elem)

def wt_number(wp):
    return len(wp.find_all('w:t'))

counts = []
for elem in z2:
    counts.append(wt_number(elem))

import matplotlib.pyplot as plt
plt.hist(counts)

lines_out = [str(elem) for elem in z2]

with open(path_xml.with_suffix('.txt'), 'w', encoding='utf8') as outfile:
    outfile.write('\n'.join(lines_out) + '\n')



# TODO liczba w:t - jak wyglada rozklad