import json
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Generator, Union, Any, Optional

import numpy as np
from py360tools import CMP, ProjectionBase
from py360tools.utils import LazyProperty
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim

from lib.assets.autodict import AutoDict
from lib.assets.chunkprojectionreader import ChunkProjectionReader
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.paths.seen_tiles_paths import SeenTilesPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.progressbar import ProgressBar
from lib.assets.worker import Worker
from lib.utils.util import save_json, load_json, get_nested_value, print_error

Tiling = str
Tile = str
NumpyArray = np.ndarray


class Props(Worker, ViewportQualityPaths, MakeDecodablePaths,
            SeenTilesPaths, ChunkQualityPaths, ABC):
    seen_tiles_deg_path: dict[Tile, Path]
    seen_tiles_ref_path: dict[Tile, Path]
    seen_tiles_result: Optional[dict]
    canvas: NumpyArray
    ui: ProgressBar
    get_tiles: dict
    proj_obj: dict[Tiling, CMP]
    results: defaultdict

    @property
    def chunk_yaw_pitch_roll_per_frame(self) -> list[tuple[float, float, float]]:
        chunk = int(self.chunk) - 1
        start = chunk * 30
        return self.user_hmd_data[slice(start, start + 30)]

    @property
    def seen_tiles(self):
        keys = [self.name, self.projection, self.tiling, self.user]
        return get_nested_value(self.seen_tiles_result, keys)['chunks'][self.chunk]

    @LazyProperty
    def hmd_dataset(self) -> dict[str, dict[str, list]]:
        return load_json(self.config.dataset_file)

    @property
    def user_hmd_data(self):
        return self.hmd_dataset[self.name + '_nas'][self.user]

    def make_data_generator(self) -> Generator[dict[str, Union[Union[ChunkProjectionReader, CMP, list[tuple[float, float, float]]], Any]], None, None]:
        """
        Generate (proj_obj: CMP,
                  frame_id: int,
                  yaw_pitch_roll: tuple[float, float, float],
                  ref_frame: ChunkProjectionReader,
                  deg_frame: ChunkProjectionReader)
        :return: None
        """
        hmd_dataset = load_json(self.config.dataset_file)

        for self.name in self.name_list:
            self.seen_tiles_result = None
            for self.user in self.users_list:
                '''
                user_hmd_data: list[list[float, float, float]]
                user_hmd_data[frame] = [yaw, pitch, roll]
                yaw, pitch, roll in radians
                '''
                user_hmd_data = None
                for self.tiling in self.tiling_list:
                    for self.chunk in self.chunk_list:
                        chunk_yaw_pitch_roll_per_frame = None
                        for self.quality in self.quality_list:
                            try:
                                if self.user_viewport_quality_json.stat().st_size == 0:
                                    self.user_viewport_quality_json.unlink()
                                    raise FileNotFoundError
                                print(f'{self.name}_{self.projection}_user{self.user}_{self.tiling}_chunk{self.chunk}_qp{self.quality}.', end='')
                                print_error(f' File exists. skipping.')
                                continue
                            except FileNotFoundError:
                                pass

                            self.seen_tiles_result = load_json(self.seen_tiles_result_json) if self.seen_tiles_result is None else self.seen_tiles_result

                            user_hmd_data = hmd_dataset[self.name + '_nas'][self.user] if user_hmd_data is None else user_hmd_data
                            chunk_yaw_pitch_roll_per_frame = user_hmd_data[(int(self.chunk) - 1) * 30: (int(self.chunk) - 1) * 30 + 30] if chunk_yaw_pitch_roll_per_frame is None else chunk_yaw_pitch_roll_per_frame
                            proj_obj = CMP(tiling=self.tiling, proj_res='3240x2160',
                                           vp_res='1320x1080', fov_res='110x90')
                            ref_tiles_path = {self.tile: self.reference_chunk for self.tile in self.seen_tiles}
                            deg_tiles_path = {self.tile: self.decodable_chunk for self.tile in self.seen_tiles}

                            data = {'proj_obj': proj_obj,
                                    'yaw_pitch_roll_by_frame': chunk_yaw_pitch_roll_per_frame,
                                    'ref_tiles_path': ref_tiles_path,
                                    'deg_tiles_path': deg_tiles_path,
                                    'user_viewport_quality_json': self.user_viewport_quality_json,
                                    }
                            yield data
                            del proj_obj

    @staticmethod
    def processa_elemento(data: dict):
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
        proj_obj: ProjectionBase = data['proj_obj']
        ref_tiles_path: dict[str, Path] = data['ref_tiles_path']
        deg_tiles_path: dict[str, Path] = data['deg_tiles_path']
        yaw_pitch_roll_by_frame: list[tuple[float, float, float]] = data['yaw_pitch_roll_by_frame']
        user_viewport_quality_json: Path = data['user_viewport_quality_json']
        results = defaultdict(list)

        ref_proj_frame_vreader = ChunkProjectionReader(ref_tiles_path, proj=proj_obj)
        deg_proj_frame_vreader = ChunkProjectionReader(deg_tiles_path, proj=proj_obj)
        ctx = f'{user_viewport_quality_json.as_posix()}'.split('/')[-5:]
        ctx[-1] = ctx[-1].replace('user_viewport_quality_', '').replace('.json', '')
        ctx[-2] = f'qp{ctx[-2]}'
        ctx[-3] = f'user{ctx[-3]}'
        ctx = '_'.join(ctx)

        for frame, yaw_pitch_roll in enumerate(yaw_pitch_roll_by_frame):
            print(f'{ctx}_frame{frame:02}/30')
            viewport_frame_ref = ref_proj_frame_vreader.extract_viewport(yaw_pitch_roll)
            viewport_frame_deg = deg_proj_frame_vreader.extract_viewport(yaw_pitch_roll)
            # Image.fromarray(viewport_frame_ref).show()
            # Image.fromarray(np.abs(viewport_frame_ref - viewport_frame_deg)).show()

            _mse = mse(viewport_frame_ref, viewport_frame_deg)
            _ssim = ssim(viewport_frame_ref, viewport_frame_deg, data_range=255.0,
                         gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            results['mse'].append(_mse)
            results['ssim'].append(_ssim)
        save_json(results, user_viewport_quality_json)


class ViewportQuality(Props):
    def init(self):
        self.projection = 'cmp'

    def main(self):
        """
        User Viewport Frame Quality
        FrameResult = (frame_id, (mse, ssim))
        self.results = list of Result [FrameResult, ...]
        :return:
        """
        # with multiprocessing.Pool() as pool:
        # pool.map(self.processa_elemento,
        #          self.make_data_generator())

        data_generator = self.make_data_generator()
        for data in data_generator:
            self.processa_elemento(data)


class CheckViewportQuality(ViewportQuality):
    def iter_name_user_tiling_chunk(self):
        self.proj_obj = {}
        for self.name in self.name_list:
            self.seen_tiles_result = load_json(self.seen_tiles_result_json)
            for self.user in self.users_list:
                for self.tiling in self.tiling_list:
                    self.proj_obj[self.projection] = CMP(tiling=self.tiling, proj_res=self.scale,
                                                         vp_res='1320x1080', fov_res=self.fov_res)
                    for self.chunk in self.chunk_list:
                        yield

    def main(self):
        resume = AutoDict()
        for _ in self.iter_name_user_tiling_chunk():
            for self.quality in self.quality_list:
                print(f'\r{self.name}_user{self.user}_{self.tiling}_chunk{self.chunk}_qp{self.quality}', end='')
                if self.user_viewport_quality_json.exists():
                    continue
                resume[self.name][f'user{self.user}'][self.tiling][f'chunk{self.chunk}'][f'qp{self.quality}'] = {}
                print_error(f'Missing')
        print(json.dumps(resume, indent=2))
        Path(f'CheckViewportQuality.json').write_text(json.dumps(resume, indent=2))


class GetViewportQuality(ViewportQuality):
    def main(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.user in self.users_list:
                    for self.tiling in self.tiling_list:
                        for self.chunk in self.chunk_list:
                            for self.quality in self.quality_list:
                                # user_viewport_quality = load_json(self.user_viewport_quality_json)

                                print(f'\r{self.name}_user{self.user}_{self.tiling}_chunk{self.chunk}_qp{self.quality}', end='')
