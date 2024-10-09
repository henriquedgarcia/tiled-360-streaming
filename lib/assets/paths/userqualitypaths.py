from pathlib import Path

import pandas as pd

from lib.assets.context import Context
from lib.assets.paths.basepaths import BasePaths
from lib.assets.ctxinterface import CtxInterface


class UserQualityPaths(CtxInterface):
    def __init__(self, context: Context):
        self.ctx = context
        self.config = context.config
        self.base_paths = BasePaths(context)

    @property
    def user_quality_folder(self) -> Path:
        folder = (self.base_paths.user_quality_folder / 'user_quality'
                  / self.name / self.projection / self.quality / self.tiling)
        return folder

    @property
    def user_metrics_json(self) -> Path:
        filename = f'seen_tiles_user{self.user}.json'
        return self.user_quality_folder / filename

    @property
    def counter_tiles_folder(self) -> Path:
        folder = (self.base_paths.user_quality_folder / 'counter_tiles'
                  / self.name / self.projection / self.quality / self.tiling)
        return folder

    @property
    def counter_tiles_json(self):
        filename = f'counter_fov{self.fov}.json'
        return self.counter_tiles_folder / filename

    @property
    def heatmap_folder(self) -> Path:
        folder = (self.base_paths.user_quality_folder / 'heatmap'
                  / self.name / self.projection / self.quality / self.tiling)
        return folder

    @property
    def heatmap_tiling(self):
        filename = f'heatmap_tiling_fov{self.config.fov}.png'
        return self.heatmap_folder / filename

    @property
    def video_test_folder(self):
        folder = (self.base_paths.user_quality_folder / 'video_test'
                  / self.name / self.projection / self.quality / self.tiling)
        return folder

    @property
    def video_test(self):
        filename = f'video_test_fov{self.config.fov}_{self.name}_{self.projection}_{self.quality}_{self.tiling}.png'
        return self.video_test_folder / filename
