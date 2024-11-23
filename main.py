#!/usr/bin/env python3
import matplotlib as mpl;

mpl.use("Qt5Agg")
import argparse
from pathlib import Path

from config.config import Config
from lib.assets.context import Context
from lib.assets.worker import Worker
from lib.decode import Decode
from lib.get_tiles import GetTiles
from lib.makedash import MakeDash
from lib.makedecodable import MakeDecodable
from lib.maketiles import MakeTiles
from lib.makequality import TileQuality
from lib.make_siti import MakeSiti
from lib.getdectime import GetDectime
from lib.getbitrate import GetBitrate
from lib.getquality import GetQuality
from lib.utils.main_utils import make_help_txt, menu, Option, get_option


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=make_help_txt(config_list, videos_list, worker_list))
    parser.add_argument('-r', default=None, nargs=3, metavar=('CONFIG_ID', 'VIDEOS_LIST_ID', 'WORKER_ID'),
                        help=f'Three string separated by space. See help text for details')
    args = parser.parse_args()

    if args.r is None:
        print(f'Choose a Config:')
        config_file = menu(config_list, init_indent=1).obj

        print(f'Choose a videos list:')
        videos_file = menu(videos_list, init_indent=1).obj

        print(f'Choose a worker:')
        worker = menu(worker_list, init_indent=1).obj
    else:
        config_id, videos_list_id, worker_id = map(int, args.r)
        config_file = get_option(config_id, config_list).obj
        videos_file = get_option(videos_list_id, videos_list).obj
        worker = get_option(worker_id, worker_list).obj

    config = Config(config_file, videos_file)
    ctx = Context(config=config)
    app: Worker = worker(ctx)
    print(f'\tTotal iterations = {app.ctx.iterations}')


path_config = Path('config')

config_list = [
    Option(id=0, name='full_qp', obj=path_config / 'config_full_qp.json'),
    Option(id=1, name='reversed_qp', obj=path_config / 'config_reversed_qp.json'),
    Option(id=2, name='full', obj=path_config / 'config_full.json'),
    Option(id=3, name='reversed', obj=path_config / 'config_reversed.json'),
    Option(id=5, name='test_qp', obj=path_config / 'config_test_qp.json'),
]

videos_list = [
    Option(id=0, name='full', obj=path_config / 'videos_full.json'),  # 
    Option(id=1, name='reversed', obj=path_config / 'videos_reversed.json'),  # 
    Option(id=2, name='lumine', obj=path_config / 'videos_lumine.json'),  # angel_falls-closet_tour
    Option(id=3, name='container0', obj=path_config / 'videos_container0.json'),  # drone_chases_car-dubstep_dance
    Option(id=4, name='container1', obj=path_config / 'videos_container1.json'),  # elevator_lift-nyc_drive
    Option(id=5, name='fortrek', obj=path_config / 'videos_fortrek.json'),  # pac_man-sunset
    Option(id=6, name='hp_elite', obj=path_config / 'videos_hp_elite.json'),  # three_peaks-video_22
    Option(id=7, name='alambique', obj=path_config / 'videos_alambique.json'),  # video_23-wingsuit_dubai
    Option(id=8, name='test', obj=path_config / 'videos_test.json'),  # 
]

worker_list = [
    Option(id=0, name='MakeTiles', obj=MakeTiles),
    Option(id=1, name='MakeDash', obj=MakeDash),
    Option(id=2, name='MakeDecodable', obj=MakeDecodable),
    Option(id=3, name='Decode', obj=Decode),
    Option(id=4, name='TileQuality', obj=TileQuality),
    Option(id=5, name='GetTiles', obj=GetTiles),
    Option(id=6, name='MakeSiti', obj=MakeSiti),
    Option(id=7, name='GetBitrate', obj=GetBitrate),
    Option(id=8, name='GetDectime', obj=GetDectime),
    Option(id=9, name='GetQuality', obj=GetQuality),
    # Option(id=10, name='GetGetTiles', obj=GetGetTiles),
]

if __name__ == '__main__':
    main()
    print(f'\nThe end.')
