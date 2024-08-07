#!/usr/bin/env python3
from pathlib import Path

from lib import menu, config
from lib import TileDecodeBenchmarkOptions

config_dict = {}
for config_file in Path('config').iterdir():
    if 'config' in config_file.name:
        path = Path('config') / config_file
        name = config_file.name.replace('config_', '')
        config_dict.update({name: path})

video_dict = {}
for videos_file in Path('config').iterdir():
    if 'videos' in videos_file.name:
        path = Path('config') / videos_file
        name = videos_file.name.replace('videos_', '')
        video_dict.update({name: path})

worker_dict = {'TileDecodeBenchmark': TileDecodeBenchmarkOptions,
               }

if __name__ == '__main__':
    print(f'Choose a Config:')
    config_file = menu(config_dict)

    print(f'Choose a video list:')
    videos_file = menu(video_dict)

    print(f'Choose a worker:')
    worker = menu(video_dict)

    config.set_config(config_file, videos_file)
    worker()
