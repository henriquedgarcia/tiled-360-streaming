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
from lib.getget_tiles import GetGetTiles
from lib.makeviewportquality import ViewportQuality
from lib.utils.main_utils import make_help_txt, menu, Option, get_option


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=make_help_txt(config_list,
                                                               videos_list,
                                                               worker_list,
                                                               names_list))
    parser.add_argument('-r', default=None, nargs=3, type=int,
                        metavar=('CONFIG_ID', 'VIDEOS_LIST_ID', 'WORKER_ID'),
                        help=f'Three string separated by space.')
    parser.add_argument('-slice', default=None, nargs=2, type=int,
                        metavar=('VIDEO_START', 'VIDEO_STOP',),
                        help=f'A int or slice of video range.')
    parser.add_argument('-tiling', default=None,
                        metavar=('TILING',),
                        help=f'Force tiling.')
    parser.add_argument('-quality', default=None,
                        metavar=('QUALITY',),
                        help=f'Force quality.')
    args = parser.parse_args()

    if args.r is None:
        print(f'Choose a Config:')
        config_file = menu(config_list, init_indent=1).obj

        print(f'Choose a videos list:')
        videos_opt = menu(videos_list, init_indent=1)
        videos_file = videos_opt.obj
        videos_list_id = videos_opt.id

        print(f'Choose a worker:')
        worker = menu(worker_list, init_indent=1).obj
    else:
        config_id, videos_list_id, worker_id = args.r
        config_file = get_option(config_id, config_list).obj
        videos_file = get_option(videos_list_id, videos_list).obj
        worker = get_option(worker_id, worker_list).obj

    config = Config(config_file, videos_file)

    if videos_list_id == 0 and args.slice is not None:
        start, stop = args.slice
        items_list = list(config.videos_dict.items())
        sliced_list = items_list[start:stop]
        config.videos_dict = dict(sliced_list)

    if videos_list_id == 0 and args.tiling is not None:
        config.tiling_list = [args.tiling]

    if videos_list_id == 0 and args.quality is not None:
        config.quality_list = [args.quality]

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
    Option(id=10, name='GetGetTiles', obj=GetGetTiles),
    Option(id=11, name='ViewportQuality', obj=ViewportQuality),
]

names = ["angel_falls", "blue_angels", "cable_cam", "chariot_race", "closet_tour", "drone_chases_car", "drone_footage", "drone_video", "drop_tower", "dubstep_dance",
         "elevator_lift", "glass_elevator", "montana", "motorsports_park", "nyc_drive", "pac_man", "penthouse", "petite_anse", "rhinos", "sunset", "three_peaks",
         "video_04", "video_19", "video_20", "video_22", "video_23", "video_24", "wingsuit_dubai"]

names_list = [Option(id=n, name=name, obj='')
              for n, name in enumerate(names)]

if __name__ == '__main__':
    main()
    print(f'\nThe end.')
