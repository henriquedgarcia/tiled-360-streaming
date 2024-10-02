from pathlib import Path

from lib.decode import Decode
from lib.get_tiles import GetTiles
from lib.getbitrate import GetBitrate
# from lib.make_siti import MakeSiti
from lib.getdectime import GetDectime
from lib.renamer_and_checker import RenamerAndCheck
from lib.segmenter import Segmenter
from lib.tilequality import TileChunkQuality

path_config = Path('config')

config_dict = {'full': path_config / 'config_full.json',  # 0
               'full_qp': path_config / 'config_full_qp.json',  # 1
               'reversed': path_config / 'config_reversed.json',  # 2
               'reversed_qp': path_config / 'config_reversed_qp.json',  # 3
               'test': path_config / 'config_test.json',  # 4
               'test_qp': path_config / 'config_test_qp.json'  # 5
               }

videos_dict = {'full': path_config / 'videos_0_full.json',  # 0
               'alambique': path_config / 'videos_alambique.json',  # 1
               'container0': path_config / 'videos_container0.json',  # 2
               'container1': path_config / 'videos_container1.json',  # 3
               'fortrek': path_config / 'videos_fortrek.json',  # 4
               'hp-elite': path_config / 'videos_hp-elite.json',  # 5
               'lumine': path_config / 'videos_lumine.json',  # 6
               'nas_cmp': path_config / 'videos_nas_cmp.json',  # 7
               'nas_erp': path_config / 'videos_nas_erp.json',  # 8
               'reversed': path_config / 'videos_reversed.json',  # 9
               'test': path_config / 'videos_test.json',  # 10
               }

worker_dict = {'Segmenter': Segmenter,  # 0
               'Decode': Decode,  # 1
               'GetTiles': GetTiles,  # 2
               'TileQuality': TileChunkQuality,  # 3
               'GetBitrate': GetBitrate,  # 4
               'GetDectime': GetDectime,  # 5
               # 'MakeSiti': MakeSiti,
               'RenamerAndCheck': RenamerAndCheck}  # 6


def make_help_txt():
    config_options = {n: k for n, k in enumerate(config_dict.keys())}
    videos_options = {n: k for n, k in enumerate(videos_dict.keys())}
    worker_options = {n: k for n, k in enumerate(worker_dict.keys())}
    text = (f'Dectime Testbed.\n'
            f'================\n'
            f'CONFIG_ID:\n'
            f'{config_options}\n'
            f'VIDEOS_LIST_ID\n'
            f'{videos_options}\n'
            f'WORKER_ID\n'
            f'{worker_options}')
    return text


def show_options(dict_options: dict, counter=0, level=0, keys_list=None, text='', mute=False, init_indent=0):
    if keys_list is None:
        keys_list = []
    for k, v in dict_options.items():
        if isinstance(v, dict):
            text += '\t' * level + f'{k}\n'
            show_options(v, counter, level + 1, keys_list, text)
        else:
            text += '\t' * (level + init_indent) + f'{counter} - {k}\n'
            keys_list.append(v)
            counter += 1
    if level == 0 and not mute:
        print(text)
    return keys_list, text


def get_options():
    try:
        chosen = int(input(f'Option: '))
    except ValueError:
        chosen = None
    return chosen


def menu(dict_options, init_indent=0):
    while True:
        keys, text = show_options(dict_options, init_indent=init_indent)
        chosen = get_options()
        if chosen is not None:
            break
    return keys[int(chosen)]
