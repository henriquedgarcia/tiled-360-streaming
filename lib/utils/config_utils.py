from pathlib import Path

from lib.segmenter import Segmenter
from lib.decode import Decode
from lib.get_tiles import GetTiles
from lib.getbitrate import GetBitrate
from lib.make_siti import MakeSiti
from lib.getdectime import GetDectime
from lib.renamer_and_checker import RenamerAndCheck

path_config = Path('config')

config_dict = {'full': path_config / 'config_full.json',
               'reversed': path_config / 'config_reversed.json',
               'test': path_config / 'config_test.json',
               'full_qp': path_config / 'config_full_qp.json',
               'test_qp': path_config / 'config_test_qp.json'
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

worker_dict = {'Segmenter': Segmenter,
               'Decode': Decode,
               'GetTiles': GetTiles,
               'GetBitrate': GetBitrate,
               'MakeSiti': MakeSiti,
               'GetDectime': GetDectime,
               'RenamerAndCheck': RenamerAndCheck}
