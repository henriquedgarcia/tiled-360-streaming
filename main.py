#!/usr/bin/env python3
import argparse
from pathlib import Path

import lib
from config.config import config
from lib.utils.util import menu

path_config = Path('config')

config_dict = {'full': path_config / 'config_full.json',
               'reversed': path_config / 'config_reversed.json',
               'test': path_config / 'config_test.json',
               'full_qp': path_config / 'config_full_qp.json'
               }

videos_dict = {'full': path_config / 'videos_0_full.json',
               'alambique': path_config / 'videos_alambique.json',
               'container0': path_config / 'videos_container0.json',
               'container1': path_config / 'videos_container1.json',
               'fortrek': path_config / 'videos_fortrek.json',
               'hp-elite': path_config / 'videos_hp-elite.json',
               'lumine': path_config / 'videos_lumine.json',
               'nas_cmp': path_config / 'videos_nas_cmp.json',
               'nas_erp': path_config / 'videos_nas_erp.json',
               'reversed': path_config / 'videos_reversed.json',
               'test': path_config / 'videos_test.json',
               }

worker_dict = {'Segmenter': lib.Segmenter,
               'Decode': lib.Decode,
               'GetTiles': lib.GetTiles,
               'GetBitrate': lib.GetBitrate,
               'MakeSiti': lib.MakeSiti,
               'GetDectime': lib.GetDectime,
               'RenamerAndCheck': lib.RenamerAndCheck}


def make_help_txt():
    text = (f'Dectime Testbed.\n'
            f'================\n'
            f'CONFIG_ID:\n'
            f'{list(config_dict)}\n'
            f'VIDEOS_LIST_ID\n'
            f'{list(videos_dict)}\n'
            f'WORKER_ID\n'
            f'{list(worker_dict)}')
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=make_help_txt())
    parser.add_argument('-r', default=None, nargs=3, metavar=('CONFIG_ID', 'VIDEOS_LIST_ID', 'WORKER_ID'),
                        help=f'Three string separated by space. See help text for details')
    args = parser.parse_args()

    if args.r is None:
        print(f'Choose a Config:')
        config_file = menu(config_dict, init_indent=1)

        print(f'Choose a videos list:')
        videos_file = menu(videos_dict, init_indent=1)

        print(f'Choose a worker:')
        worker = menu(worker_dict, init_indent=1)
    else:
        config_id, videos_list_id, worker_id = map(int, args.r)

        config_file = config_dict[list(config_dict)[config_id]]
        videos_file = videos_dict[list(videos_dict)[videos_list_id]]
        worker = worker_dict[list(worker_dict)[worker_id]]

    config.set_config(config_file, videos_file)
    worker()

    print(f'\nThe end.')
