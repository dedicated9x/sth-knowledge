import sys
import argparse

def parse_arguments(input_):
    # We usually don't need path to script.
    input_ = input_[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('argname1', type=str)
    parser.add_argument('argname2', type=str)
    return parser.parse_args(input_)

sys.argv = [
    'path_to_script',
    'arg1',
    'arg2'
]

kwargs_nspace: argparse.Namespace = parse_arguments(sys.argv)
kwargs: dict = vars(kwargs_nspace)


