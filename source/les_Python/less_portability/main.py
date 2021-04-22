import pathlib as pl
import shutil
import os

class Src:
    def __init__(self, path_temp):
        self.path_temp = path_temp
        self.path_root = pl.Path(self.path_temp).joinpath('portability')

    def __repr__(self):
        # files = list(self.path_root.glob('**/*'))
        # for f in files:
        #     print(f)
        #     if f.is_file():
        #         print(f.read_text())
        # TODO to powinna być oddzielna klasa
        repr_ = []
        files = list(self.path_root.glob('**/*'))
        for f in files:
            repr_.append(f.__str__())
            if f.is_file():
                repr_.append(f.read_text())

        return '\n'.join(repr_)

    def create(self):
        path_src = self.path_root.joinpath('src')
        path_lib1 = path_src.joinpath('lib1.py')
        path_script1 = path_src.joinpath('script1.py')

        body_lib1 = """
        def func1():
            print('Execution of func1')
        """

        body_script1 = """
        from lib1 import func1
        func1()
        """

        self.path_root.mkdir()
        path_src.mkdir()
        path_lib1.write_text(body_lib1)
        path_script1.write_text(body_script1)


    def clean(self):
        shutil.rmtree(self.path_root)


path_root = rf"C:\temp"
src = Src(path_root)
print(src)

# src.create()
# src.clean()


# self = src
#
# repr_ = []
# files = list(self.path_root.glob('**/*'))
# for f in files:
#     repr_.append(f.__str__())
#     if f.is_file():
#         repr_.append(f.read_text())
#
# return '\n'.join(repr_)