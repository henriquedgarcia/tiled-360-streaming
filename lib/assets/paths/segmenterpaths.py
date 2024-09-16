from pathlib import Path

from lib.assets.context import ctx
from lib.assets.paths.basepaths import base_paths


class SegmenterPaths:
    def ___segments_files___(self):
        ...

    @property
    def lossless_video(self) -> Path:
        return base_paths.lossless_folder / ctx.projection / f'{ctx.name}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_video.with_suffix('.log')

    @property
    def tile_video(self) -> Path:
        return base_paths.tiles_folder / base_paths.basename1 / f'tile{ctx.tile}.mp4'

    @property
    def tile_log(self) -> Path:
        return self.tile_video.with_suffix('.log')

    @property
    def segmenter_log(self) -> Path:
        return base_paths.segmenter_folder / base_paths.basename1 / f'tile{ctx.tile}_segmenter.log'

    @property
    def chunks_folder(self) -> Path:
        return base_paths.segmenter_folder / base_paths.basename2

    @property
    def chunk_video(self) -> Path:
        chunk = int(ctx.chunk)
        return self.chunks_folder / f'tile{ctx.tile}_{chunk:03d}.hevc'

    @property
    def bitrate_result_json(self) -> Path:
        return base_paths.results_folder / f'rate_{ctx.name}.json'


segmenter_paths = SegmenterPaths()
