#!/usr/bin/env python3

# mpl.use("Qt5Agg")
import argparse
from pathlib import Path

from config.config import Config
from lib.assets.context import Context
from lib.assets.worker import Worker
from lib.check import Check
from lib.make_dectime import MakeDectime
from lib.get_bitrate import GetBitrate
from lib.get_chunk_quality import GetChunkQuality
from lib.get_seen_tiles import GetTilesSeen
from lib.make_tiles_seen import MakeTilesSeen
from lib.get_dectime import GetDectime
from lib.make_siti import MakeSiti, GetMakeSiti
from lib.make_dash import MakeDash
from lib.make_decodable import MakeDecodable
from lib.make_chunk_quality import MakeChunkQuality
from lib.make_tiles import MakeTiles
from lib.make_viewport_quality import ViewportQuality, CheckViewportQuality
from lib.get_viewport_quality import GetViewportQuality
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
    parser.add_argument('-video', default=None,
                        metavar=('VIDEO',),
                        help=f'Force video. [0, 1, 2, 3, 4, 5, 6, 7]')
    parser.add_argument('-projection', default=None,
                        metavar=('PROJECTION',),
                        help=f"Force projection. ['cmp' | 'erp]")
    parser.add_argument('-tiling', default=None,
                        metavar=('TILING',),
                        help=f'Force tiling. ["1x1", "3x2", "6x4", "9x6", "12x8"]')
    parser.add_argument('-quality', default=None,
                        metavar=('QUALITY',),
                        help=f'Force quality.')
    parser.add_argument('-qslice', default=None, nargs=2, type=int,
                        metavar=('QUALITY_START', 'QUALITY_STOP',),
                        help=f'Force quality. ["16", "22", "28", "34", "40", "46"]')
    parser.add_argument('-remove', default=0,
                        metavar=('QUALITY',),
                        help=f'Force remove.')
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
    config.remove = bool(args.remove)

    if videos_list_id in [0, 9] and args.slice is not None:
        start, stop = args.slice
        items_list = list(config.videos_dict.items())
        sliced_list = items_list[start:stop]
        config.videos_dict = dict(sliced_list)

    if args.video is not None:
        video = args.video
        items_list = list(config.videos_dict.items())
        selected_video = items_list[int(video)]
        config.videos_dict = {selected_video[0]: selected_video[1]}

    if args.projection is not None:
        config.projection_list = [args.projection]

    if args.tiling is not None:
        config.tiling_list = [args.tiling]

    if args.quality is not None:
        config.quality_list = [args.quality]

    if args.qslice is not None:
        start, stop = args.qslice
        config.quality_list = config.quality_list[start:stop]

    ctx = Context(config=config)
    app: Worker = worker(ctx)
    app.run()
    print(f'\tTotal iterations = {app.ctx.iterations}')


path_config = Path('config')

config_list = [
    Option(id=0, name='cmp_qp', obj=path_config / 'config_cmp_qp.json'),
    Option(id=1, name='erp_qp', obj=path_config / 'config_erp_qp.json'),
    Option(id=2, name='cmp_crf', obj=path_config / 'config_cmp_crf.json'),
    Option(id=3, name='erp_crf', obj=path_config / 'config_erp_crf.json'),
]

videos_list = [
    Option(id=0, name='reduced', obj=path_config / 'videos_reduced.json'),  #
    Option(id=1, name='reversed', obj=path_config / 'videos_reversed.json'),  #
    Option(id=2, name='lumine', obj=path_config / 'videos_lumine.json'),  # 100 angel_falls-closet_tour
    Option(id=3, name='container0', obj=path_config / 'videos_container0.json'),  # 67 drone_chases_car-dubstep_dance
    Option(id=4, name='container1', obj=path_config / 'videos_container1.json'),  # 70 elevator_lift-nyc_drive
    Option(id=5, name='fortrek', obj=path_config / 'videos_fortrek.json'),  # 101 pac_man-sunset
    Option(id=6, name='hp_elite', obj=path_config / 'videos_hp_elite.json'),  # 103 three_peaks-video_22
    Option(id=7, name='alambique', obj=path_config / 'videos_alambique.json'),  # 99 video_23-wingsuit_dubai
    Option(id=8, name='test', obj=path_config / 'videos_test.json'),  # 
    Option(id=9, name='full', obj=path_config / 'videos_full.json'),  #
]

worker_list = [
    Option(id=0, name='MakeTiles', obj=MakeTiles),
    Option(id=1, name='MakeDash', obj=MakeDash),
    Option(id=2, name='MakeDecodable', obj=MakeDecodable),
    Option(id=3, name='MakeDectime', obj=MakeDectime),
    Option(id=4, name='MakeChunkQuality', obj=MakeChunkQuality),
    Option(id=5, name='MakeTilesSeen', obj=MakeTilesSeen),
    Option(id=6, name='MakeSiti', obj=MakeSiti),
    Option(id=7, name='GetBitrate', obj=GetBitrate),
    Option(id=8, name='GetDectime', obj=GetDectime),
    Option(id=9, name='GetChunkQuality', obj=GetChunkQuality),
    Option(id=10, name='GetTilesSeen', obj=GetTilesSeen),
    Option(id=11, name='ViewportQuality', obj=ViewportQuality),
    Option(id=12, name='CheckViewportQuality', obj=CheckViewportQuality),
    Option(id=13, name='GetViewportQuality', obj=GetViewportQuality),
    Option(id=14, name='GetMakeSiti', obj=GetMakeSiti),
    Option(id=99, name='Check', obj=Check),
]

names = ["cable_cam", "closet_tour", "drop_tower", "glass_elevator", "pac_man", "penthouse", "sunset", "wingsuit_dubai"]

names_list = [Option(id=n, name=name, obj='')
              for n, name in enumerate(names)]

if __name__ == '__main__':
    main()
    print(f'\nThe end.')
