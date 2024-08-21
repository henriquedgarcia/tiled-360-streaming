#!/usr/bin/env python3
import argparse

from config.config import config
from lib.utils.util import menu
from lib.utils.config_utils import config_dict, videos_dict, worker_dict


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
