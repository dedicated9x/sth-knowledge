"""pozbycie sie gowno znakow"""
# ban_list = str.maketrans(
#     "ąęóćźżńłśĆŹŻŃŁŚ",
#     "               "
# )
# content = 'None'
# content.translate(ban_list)


# import os
# import gc

import pathlib as pl
from wand.image import Image as wi
import io
import pandas as pd
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\devoted\Desktop\Tesseract-OCR\tesseract.exe'



def path2blobs(path_):
    pdf = wi(filename=path_, resolution=300)
    pdfImg = pdf.convert('jpeg')
    imgBlobs = []

    # CT_REM = 0 # TODO REMOVE
    for img in pdfImg.sequence:
        page = wi(image=img)
        imgBlobs.append(page.make_blob('jpeg'))

        # CT_REM += 1 # TODO REMOVE
        # if CT_REM >= 3: # TODO REMOVE
        #     break # TODO REMOVE
    return imgBlobs

def blob2text(blob):
    im = Image.open(io.BytesIO(blob))
    text = pytesseract.image_to_string(im, lang='deu')
    return text


"""MAIN"""
# pdf_path = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\chapter_corr\4.pdf")
# out_path = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\out\test_de.jpg")

df_aslist = []

src_path = pl.Path(rf"C:\Users\devoted\Desktop\ksiazki_nat_fp\chapter_corr")
pdf_list = list(src_path.glob('**/*'))
# pdf_list = pdf_list[:2] # TODO REMOVE

pdf_path = pdf_list[0]

for pdf_path in pdf_list:
    # PDF_NO = pdf_path.stem

    blobs = path2blobs(pdf_path)
    texts = [blob2text(b) for b in blobs]

    for idx, text in enumerate(texts):
        df_aslist.append({
            'CHAPTER': int(pdf_path.stem),
            'PAGE': idx + 1,
            'TEXT': text
        })


df = pd.DataFrame(df_aslist)
path_csv = pl.Path(__file__).parent.joinpath('static', 'book_ocred.csv')
path_excel = path_csv.with_suffix('.xlsx')
df.to_csv(path_csv)
df.to_excel(path_excel)

# TODO 1 paths
# TODO 2 pandas (CHAPTER, PAGE, TEXT)
#
# for idx, text in enumerate(texts):
#     print(idx+1)


"""de - new"""
# im = Image.open(out_path)
# im.load()
# text = pytesseract.image_to_string(im, lang='deu')

"""extract"""
# extracted_text = []
# for imgBlob in imgBlobs[:2]:
#     im = Image.open(io.BytesIO(imgBlob))
#     text = pytesseract.image_to_string(im, lang='eng')
#     extracted_text.append(text)


"""z wiersza polecen"""
# "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" test_de.jpg out -l deu


"""save"""
# with open(out_path, 'wb') as outfile:
#     outfile.write(imgBlobs2[2])

