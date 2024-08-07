from pathlib import Path

from .config import config
from .context import ctx


class Paths:
    project_path = Path('../results') / config.project_folder
    lossless_folder = project_path / 'lossless'
    segments_folder = project_path / 'segments'
    quality_folder = project_path / 'quality'

    graphs_folder = project_path / 'graphs'
    viewport_folder = project_path / 'viewport'
    siti_folder = project_path / 'siti'

    @property
    def basename1(self):
        return (Path(f'{ctx.name}') /
                f'{ctx.proj}' /
                f'{config.rate_control}{ctx.quality}' /
                f'{ctx.tiling}'
                )

    @property
    def basename2(self):
        return self.basename1 / f'tile{ctx.tile}'

    def ___segments_files___(self): ...

    @property
    def lossless_file(self) -> Path:
        return self.lossless_folder / ctx.projection / f'{ctx.name}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_file.with_suffix('.log')

    @property
    def compressed_file(self) -> Path:
        return self.segments_folder / self.basename1 / f'tile{ctx.tile}.mp4'

    @property
    def compressed_log(self) -> Path:
        return self.compressed_file.with_suffix('.log')

    @property
    def segmenter_log(self) -> Path:
        return self.compressed_file.with_name(f'tile{ctx.tile}_segmenter.log')

    @property
    def segment_file(self) -> Path:
        chunk = int(ctx.chunk)
        return self.segments_folder / self.basename2 / f'tile{ctx.tile}_{chunk:03d}.mp4'

    @property
    def dectime_log(self) -> Path:
        chunk = int(str(ctx.chunk))
        return self.segment_file.with_name(f'tile{ctx.tile}_{chunk:03d}_dectime.log')

    @property
    def reference_segment(self):
        qlt = ctx.quality
        ctx.quality = config.original_quality
        segment_file = self.segment_file
        ctx.quality = qlt
        return segment_file

    def ___json_results_files___(self): ...

    @property
    def dectime_result_json(self) -> Path:
        return self.segments_folder / f'time_{ctx.name}.json'

    @property
    def bitrate_result_json(self) -> Path:
        return self.segments_folder / f'rate_{ctx.name}.json'

    @property
    def quality_result_json(self) -> Path:
        return self.segments_folder / f'quality_{ctx.name}.json'
    # Tiles chunk path


paths = Paths()
