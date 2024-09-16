#!/usr/bin/env python3
import argparse

from config.config import Config
from lib.assets.context import Context
from lib.utils.main_utils import config_dict, videos_dict, worker_dict, make_help_txt, menu

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

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    worker(config, ctx)

    print(f'\nThe end.')
