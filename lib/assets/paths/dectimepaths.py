from pathlib import Path

from lib.assets.context import ctx
from lib.assets.paths import segmenter_paths, base_paths


class DectimePaths:
    @property
    def dectime_log(self) -> Path:
        chunk = int(str(ctx.chunk))
        return segmenter_paths.chunk_video.with_name(f'tile{ctx.tile}_{chunk:03d}_dectime.log')

    @property
    def dectime_result_json(self) -> Path:
        return base_paths.results_folder / f'time_{ctx.name}.json'


dectime_paths = DectimePaths()
