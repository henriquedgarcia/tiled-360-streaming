from pathlib import Path
from typing import Union

from lib.assets import Factors


class GlobalPaths(Factors):
    overwrite = False
    graphs_folder = Path('graphs')
    viewport_folder = Path('viewport')
    quality_folder = Path('quality')
    siti_folder = Path('siti')
    check_folder = Path('check')
    operation_folder = Path('')

    @property
    def project_path(self) -> Path:
        return Path('../results') / self.config['project']

    @property
    def basename(self):
        return Path(f'{self.proj}/'
                    f'{self.name}/'
                    f'{self.tiling}/'
                    f'{self.rate_control}{self.quality}/'
                    f'{self.tile}')

    @property
    def dectime_result_json(self) -> Path:
        """
        By Video
        :return:
        """
        folder = self.project_path / self.dectime_folder
        folder.mkdir(parents=True,
                     exist_ok=True)
        return folder / f'time_{self.video}.json'

    @property
    def bitrate_result_json(self) -> Path:
        folder = self.project_path / self.segment_folder
        folder.mkdir(parents=True,
                     exist_ok=True)
        return folder / f'rate_{self.video}.json'

    @property
    def quality_result_json(self) -> Path:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True,
                     exist_ok=True)
        return folder / f'quality_{self.video}.json'
    # Tiles chunk path
