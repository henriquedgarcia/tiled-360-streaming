#!/usr/bin/env python3
from pathlib import Path
import argparse
from typing import Type
import lib
from lib import TileDecodeBenchmarkOptions

config_dict = {'full': 'config_full.py',
               'reversed': 'config_reversed.py'
               }

video_dict = {'full': 'videos_full.py',
              'reversed': 'videos_reversed.py',
              'alambique': 'videos_alambique.py',
              'container0': 'videos_container0.py',
              'container1': 'videos_container1.py',
              'fortrek': 'videos_fortrek.py',
              'hp - elite': 'videos_hp - elite.py',
              'lumine': 'videos_lumine.py',
              'only cmp': 'videos_28videos_nas_cmp.py',
              'only erp': 'videos_28videos_nas_erp.py'
              }

worker_dict = {'TileDecodeBenchmark': TileDecodeBenchmarkOptions,
              }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=make_help_txt())
    parser.add_argument('-c', default=None, metavar='CONFIG_FILE', help='The path to config file')
    parser.add_argument('-r', default=None, nargs=2, metavar=('WORKER_ID', 'ROLE_ID'), help=f'Two int separated by space.')
    args = parser.parse_args()

    print(f'Choose a Config:')
    keys = list(config_dict.keys())
    while True:
        for n, k in enumerate(keys):
            print(f'\t{n} - {k}')
        try:
            option = int(input(f'Option: '))
        except ValueError:
            continue
        break
    option_key = keys[option]
    file = Path('config') / config_dict[option_key]
    config = eval(file.read_text())

    print(f'Choose a video list:')
    keys = list(video_dict.keys())
    while True:
        for n, k in enumerate(keys):
            print(f'\t{n} - {k}')
        try:
            option = int(input(f'Option: '))
        except ValueError:
            continue
        break
    option_key = keys[option]
    file = Path('config') / config_dict[option_key]
    config['videos'] = eval(file.read_text())

    print(f'Choose a worker:')
    items = list(worker_dict.items())
    n=0
    list_temp = []
    while True:
        for k, worker in items:
            print(f'{k}:')
            for role  in worker.values():
                print(f'\t{n} - {role.__name__}')
                list_temp.append(role)
                n+=1
        try:
            option = int(input(f'Option: '))
            list_temp[option](config)
        except ValueError:
            continue
        break
