from pathlib import Path

from lib.assets.paths.make_decodable_paths import MakeDecodablePaths


class DectimePaths(MakeDecodablePaths):
    @property
    def dectime_folder(self) -> Path:
        return self.dectime_folder0 / self.folder_name_proj_tiling_tile_qlt

    @property
    def dectime_log(self) -> Path:
        chunk = int(self.chunk)
        return self.dectime_folder / f'chunk{chunk:03d}_dectime.log'

    @property
    def dectime_result_json(self) -> Path:
        return self.results_folder / f'dectime/time_{self.ctx.name}.json'

    @property
    def dectime_result_by_name(self) -> Path:
        return self.dectime_folder0 / f'dectime_{self.name}_{self.projection}_{self.rate_control}.pickle'

    @property
    def dectime_result(self) -> Path:
        return self.results_folder / f'dectime_{self.projection}_{self.rate_control}.pickle'
