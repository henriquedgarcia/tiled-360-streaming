from pathlib import Path
from typing import Union

from util import load_json
from .context_lists import ContextLists
from .factors import Factors


class Config(ContextLists, Factors):
    config_dict: dict[str, Union[str, int, dict, list]]

    def set_config(self, config_file, videos_file):
        self.config_dict = load_json(config_file)
        self._videos_dict = load_json(videos_file)
        self._dataset = load_json(self.config_dict['dataset_file'])
        self._sph_file = load_json(self.config_dict['sph_file'])



config = Config()
