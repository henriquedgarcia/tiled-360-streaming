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
    def dectime_result_pickle(self) -> Path:
        return self.results_folder / f'dectime/dectime.pickle'
