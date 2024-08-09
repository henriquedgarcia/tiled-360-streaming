#!/usr/bin/env python3
from pathlib import Path

import lib
from config.config import config
from lib.utils.util import menu

config_dict = {}
for config_file in Path('config').iterdir():
    if 'config_' in config_file.name:
        name = config_file.name.replace('config_', '')
        config_dict.update({name: config_file})

video_dict = {}
for videos_file in Path('config').iterdir():
    if 'videos' in videos_file.name:
        name = videos_file.name.replace('videos_', '')
        video_dict.update({name: videos_file})

worker_dict = {getattr(lib, worker).__name__: getattr(lib, worker)
               for worker in lib.__all__}

if __name__ == '__main__':
    print(f'Choose a Config:')
    config_file = menu(config_dict)

    print(f'Choose a video list:')
    videos_file = menu(video_dict)

    print(f'Choose a worker:')
    worker = menu(worker_dict)

    config.set_config(config_file, videos_file)
    worker()
