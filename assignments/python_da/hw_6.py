import pathlib as pl
import random
import itertools
import os

def generate_body():
    return " Model; Output value; Time of computation;\n {}; {} ; {}s;".format(
        random.choice(['A', 'B', 'C']),
        random.randint(0, 1000),
        random.randint(0, 1000)
    )

path_root = pl.Path(os.getcwd())
# path_root = pl.Path(rf"C:\temp")
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
timesofday = ['morning', 'evening']

# PART A
files = [path_root.joinpath(d, t, 'Solutions.csv') for d, t in itertools.product(weekdays, timesofday)]
for f in files:
    try:
        f.parent.parent.mkdir()
    except FileExistsError:
        pass
    f.parent.mkdir()
    with open(f, 'w') as outfile:
        outfile.write(generate_body())

# PART B
list(path_root.glob('**/*.csv'))
data = [p.read_text().split('\n')[1].split(';') for p in path_root.glob('**/*.csv')]
times_A = [d[2] for d in data if d[0] == ' A']
result = sum([int(t[:-1]) for t in times_A])
print(f'{result}s')
