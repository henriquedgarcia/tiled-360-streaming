#!/usr/bin/env python3
from pathlib import Path

from lib import TileDecodeBenchmarkOptions, load_json
from lib.util import menu

config_dict = {config_file.name.replace('config_', ''): config_file
               for config_file in Path('config').iterdir() if 'config' in config_file.name}
video_dict = {videos_file.name.replace('videos_', ''): videos_file
              for videos_file in Path('config').iterdir() if 'videos' in videos_file.name}

worker_dict = {'TileDecodeBenchmark': TileDecodeBenchmarkOptions,
               }

if __name__ == '__main__':
    print(f'Choose a Config:')
    config_file = menu(config_dict)
    config = Path('config') / config_dict[config_file]

    print(f'Choose a video list:')
    videos_file = menu(video_dict)
    videos = Path('config') / videos_file

    print(f'Choose a worker:')
    worker = menu(video_dict)
    worker(config, videos)
