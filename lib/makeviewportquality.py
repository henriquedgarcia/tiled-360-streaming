from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Generator, Union, Any

import numpy as np
from py360tools import CMP
from py360tools.utils import LazyProperty
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.paths.seen_tiles_paths import SeenTilesPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.videoprojectionreader import VideoProjectionReader
from lib.assets.worker import Worker
from lib.utils.util import save_json, load_json, get_nested_value, print_error

Tiling = str
Tile = str
NumpyArray = np.ndarray


class Props(Worker, ViewportQualityPaths, MakeDecodablePaths,
            SeenTilesPaths, ChunkQualityPaths, ABC):
    seen_tiles_deg_path: dict[Tile, Path]
    seen_tiles_ref_path: dict[Tile, Path]
    canvas: NumpyArray
    ui: ProgressBar
    get_tiles: dict
    proj_obj: dict[Tiling, CMP]
    results: defaultdict
    hmd_dataset: dict[str, dict[str, list]]
    seen_tiles_result: dict

    @property
    def yaw_pitch_roll_by_frame(self) -> list[tuple[float, float, float]]:
        chunk = int(self.chunk) - 1
        start = chunk * 30
        return self.user_hmd_data[slice(start, start + 30)]

    @property
    def seen_tiles(self):
        keys = [self.name, self.projection, self.tiling, self.user]
        return get_nested_value(self.seen_tiles_result, keys)['chunks'][self.chunk]

    @LazyProperty
    def hmd_dataset(self):
        return load_json(self.config.dataset_file)

    @property
    def user_hmd_data(self):
        return self.hmd_dataset[self.name + '_nas'][self.user]

    def make_ref_vreader(self) -> VideoProjectionReader:
        """
        need self.tile, self.seen_tiles,
        self.reference_chunk,
        self.ctx (to VideoProjectionReader)
        :return:
        """
        ref_tiles_path = {}
        for self.tile in self.seen_tiles:
            if not self.reference_chunk.exists():
                raise FileNotFoundError
            ref_tiles_path[self.tile] = self.reference_chunk

        return VideoProjectionReader(ref_tiles_path, projection=self.projection,
                                     tiling=self.tiling,
                                     proj_res=self.config.proj_res,
                                     fov_res=self.config.fov_res,
                                     vp_res='1320x1080')

    def make_deg_vreader(self) -> VideoProjectionReader:
        deg_tiles_path = {}
        for self.tile in self.seen_tiles:
            if not self.decodable_chunk.exists(): raise FileNotFoundError
            deg_tiles_path[self.tile] = self.decodable_chunk

        return VideoProjectionReader(deg_tiles_path,
                                     tiling=self.tiling, proj=self.)

    def make_data_generator(self,
                            proj_obj,
                            yaw_pitch_roll_by_frame) -> Generator[dict[str, Union[Union[VideoProjectionReader, CMP, list[tuple[float, float, float]]], Any]], None, None]:
        """
        Generate (proj_obj: CMP,
                  frame_id: int,
                  yaw_pitch_roll: tuple[float, float, float],
                  ref_frame: VideoProjectionReader,
                  deg_frame: VideoProjectionReader)
        :return: None
        """
        data = {}

        self.results = defaultdict(list)

        ref_proj_frame_vreader = self.make_ref_vreader()
        for self.quality in self.quality_list:
            deg_proj_frame_vreader = self.make_deg_vreader()

            data['ref_proj_frame_vreader'] = ref_proj_frame_vreader
            data['deg_proj_frame_vreader'] = deg_proj_frame_vreader
            data['proj_obj'] = self.proj_obj
            data['yaw_pitch_roll_by_frame'] = self.yaw_pitch_roll_by_frame
            data['user_viewport_quality_json'] = self.user_viewport_quality_json
            yield data

    def processa_elemento(self, data: dict):
        user_viewport_quality_json = data['user_viewport_quality_json']
        if user_viewport_quality_json.exists(): return

        results = defaultdict(list)

        ref_proj_frame_vreader: VideoProjectionReader = data['ref_proj_frame_vreader']
        deg_proj_frame_vreader = data['deg_proj_frame_vreader']
        yaw_pitch_roll_by_frame = data['yaw_pitch_roll_by_frame']

        for frame in range(30):
            # expand data and get projection
            proj_obj = ref_proj_frame_vreader.proj
            yaw_pitch_roll = data['yaw_pitch_roll']
            ref_frame = ref_proj_frame_vreader.get_frame()
            deg_frame = deg_proj_frame_vreader.get_frame()

            # get_vp_frame
            ref_frame = ref_proj_frame_vreader.proj.extract_viewport(ref_frame, yaw_pitch_roll)
            viewport_frame_ref = proj_obj.extract_viewport(ref_frame, yaw_pitch_roll)
            viewport_frame_deg = proj_obj.extract_viewport(deg_frame, yaw_pitch_roll)
            # Image.fromarray(viewport_frame_ref).show()
            # Image.fromarray(np.abs(viewport_frame_ref - viewport_frame_deg)).show()

            # calc error
            _mse = mse(viewport_frame_ref, viewport_frame_deg)
            _ssim = ssim(viewport_frame_ref, viewport_frame_deg, data_range=255.0,
                         gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        save_json(results, user_viewport_quality_json)


class ViewportQuality(Props):
    def init(self):
        self.projection = 'cmp'

    def iter_name_user_tiling_chunk(self):
        self.proj_obj = {}
        for self.name in self.name_list:
            self.seen_tiles_result = load_json(self.seen_tiles_result_json)
            for self.user in self.users_list:
                for self.tiling in self.tiling_list:
                    self.proj_obj[self.projection] = CMP(tiling=self.tiling, proj_res=self.config.proj_res,
                                               vp_res='1320x1080', fov_res=self.config.fov_res)
                    for self.chunk in self.chunk_list:
                        yield

    def main(self):
        """
        User Viewport Frame Quality
        FrameResult = (frame_id, (mse, ssim))
        self.results = list of Result [FrameResult, ...]
        :return:
        """
        for _ in self.iter_name_user_tiling_chunk():
            self.worker()

    def worker(self):
        """
        criar tabela
        criar o vreader e yaw_pitch_roll
        Para cada frame no vreader
            reconstruir o quadro da projeção de referência e degradada conforme o viewport
            extrair viewport
            calcular erro
            adicionar erro à tabela
        salvar tabela
        :return:
        """

        data_generator = self.make_data_generator()
        self.processa_elemento(next(data_generator))
        # with multiprocessing.Pool(1) as pool:
        #     resultados = pool.map(self.processa_elemento,
        #                           self.data_generator())

        results = self.processa_elemento(next(data_generator))

        for self.quality in self.quality_list:
            data_generator = self.make_data_generator(ref_proj_frame_vreader,
                                                      deg_proj_frame_vreader,
                                                      proj_obj,
                                                      yaw_pitch_roll_by_frame)

            results = sorted(results, key=lambda value: value[0])
            list(map(lambda x: x[1], results))
            results['mse'].append(_mse)
            results['ssim'].append(_ssim)


class CheckViewportQuality(ViewportQuality):
    def init(self):
        pass

    def main(self):
        total = len(self.tiling_list) * 30 * len(self.quality_list) * 60
        ui = ProgressBar(total=total, desc=self.__class__.__name__)
        for _ in self.iter_name_user_tiling_chunk():
            ui.update(f'{self}')
            if self.user_viewport_quality_json.exists(): continue
            if self.chunk == '1':
                print_error(f'Missing {self.name}_user{self.user}_{self.tiling}_chunk{self.chunk}\n\t{self.user_viewport_quality_json}')
