from tika import parser
raw = parser.from_file("path to pdf")
print(raw['content'])