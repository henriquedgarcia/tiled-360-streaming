from pathlib import Path

from config.config import config
from .context import ctx


class Paths:
    @property
    def project_path(self):
        return config.project_folder

    @property
    def results_folder(self):
        return self.project_path / 'results_json'

    @property
    def lossless_folder(self):
        return self.project_path / 'lossless'

    @property
    def segments_folder(self):
        return self.project_path / 'segments'

    @property
    def compresseds_folder(self):
        return self.project_path / 'compresseds'

    @property
    def quality_folder(self):
        return self.project_path / 'quality'

    @property
    def graphs_folder(self):
        return self.project_path / 'graphs'

    @property
    def viewport_folder(self):
        return self.project_path / 'viewport'

    @property
    def siti_folder(self):
        return self.project_path / 'siti'

    @property
    def basename1(self):
        return (Path(f'{ctx.name}') /
                f'{ctx.projection}' /
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
        return self.compresseds_folder / self.basename1 / f'tile{ctx.tile}.mp4'

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
        rate_control = config.rate_control

        config.rate_control = 'crf'
        ctx.quality = '0'

        segment_file = self.segment_file

        ctx.quality = qlt
        ctx.rate_control = rate_control
        return segment_file

    def ___json_results_files___(self): ...

    @property
    def dectime_result_json(self) -> Path:
        return self.results_folder / f'time_{ctx.name}.json'

    @property
    def bitrate_result_json(self) -> Path:
        return self.results_folder / f'rate_{ctx.name}.json'

    @property
    def quality_result_json(self) -> Path:
        return self.results_folder / f'quality_{ctx.name}.json'

    # Tiles chunk path

    def ___siti_files___(self): ...

    @property
    def siti_stats(self) -> Path:
        self.siti_folder.mkdir(exist_ok=True, parents=True)
        return self.siti_folder / f'siti_stats.csv'

    @property
    def siti_plot(self) -> Path:
        self.siti_folder.mkdir(exist_ok=True, parents=True)
        return self.siti_folder / f'siti_plot.png'

    @property
    def siti_results(self) -> Path:
        self.siti_folder.mkdir(exist_ok=True, parents=True)
        name = f'siti_results'

        if ctx.name is not None:
            name += f'_{ctx.name}'
        if ctx.tiling is not None:
            name += f'_{ctx.tiling}'
        if ctx.quality is not None:
            name += f'_{config.rate_control}{ctx.quality}'
        if ctx.tile is not None:
            name += f'_tile{ctx.tile}'
        if ctx.chunk is not None:
            name += f'_chunk{ctx.chunk}'
        return self.siti_folder / f'{name}.csv'


paths = Paths()
