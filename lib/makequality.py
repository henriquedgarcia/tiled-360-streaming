import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import save_json, load_json, iter_video


class TileQuality(Worker, CtxInterface):
    quality_metrics: QualityMetrics
    chunk_quality_paths: ChunkQualityPaths

    def main(self):
        self.init()
        # self.check()

        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self):
                self.work()

    def check(self):
        check_data = defaultdict(list)

        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            msg = f'{self.name}_{self.projection}_{self.tiling}_tile{self.tile}_qp{self.quality}'
                            print(f'\r{msg}', end='')
                            try:
                                for self.chunk in self.chunk_list:
                                    if not self.chunk_quality_paths.chunk_quality_json.exists(): raise FileNotFoundError
                                    # size = self.chunk_quality_paths.chunk_quality_json.stat().st_size
                                    # if not size > 0: raise ValueError
                            except FileNotFoundError:
                                print(f'\n\tFileNotFoundError')
                                check_data['not_exist'].append(msg)
                            except ValueError:
                                print(f'\n\t0_size')
                                check_data['0_size'].append(msg)

        print(json.dumps(check_data, indent=2))
        Path('chunk_quality_errors.json').write_text(json.dumps(check_data, indent=2))

    def init(self):
        self.chunk_quality_paths = ChunkQualityPaths(self.ctx)
        self.quality_metrics = QualityMetrics(self.ctx)

    def work(self):
        if self.chunk_quality_json_ok():
            raise AbortError('chunk_quality_json is ok')

        reference_frames = iter_video(self.reference_chunk)
        tile_frame = iter_video(self.decodable_chunk)
        chunk_quality = defaultdict(list)
        error = []
        frame1 = frame2 = np.array([])

        for _ in tqdm(range(self.gop), desc='Chunk frame'):
            try:
                frame1 = next(reference_frames)
            except (StopIteration, ValueError):
                error.append('reference_frames error')
                # self.reference_chunk.unlink(missing_ok=True)

            try:
                frame2 = next(tile_frame)
            except (StopIteration, ValueError):
                error.append('tile_frame error')
                # self.chunk_video.unlink(missing_ok=True)

            if error:
                self.logger.register_log(', '.join(error), self.decodable_chunk)
                return

            chunk_quality['ssim'].append(self.quality_metrics.ssim(frame1, frame2))
            chunk_quality['mse'].append(self.quality_metrics.mse(frame1, frame2))
            chunk_quality['s-mse'].append(self.quality_metrics.smse_nn(frame1, frame2))
            chunk_quality['ws-mse'].append(self.quality_metrics.wsmse(frame1, frame2))
        save_json(chunk_quality, self.chunk_quality_json)

    def chunk_quality_json_ok(self) -> bool:
        try:
            chunk_quality = load_json(self.chunk_quality_paths.chunk_quality_json)
            if len(chunk_quality['mse']) != int(self.gop):
                self.chunk_quality_json.unlink(missing_ok=True)
                self.logger.register_log(f'MISSING_FRAMES', self.chunk_quality_json)
                raise FileNotFoundError()
        except FileNotFoundError:
            self.assert_chunk_and_reference()
            return False
        return True  # All OK

    def assert_chunk_and_reference(self):
        error = []
        if not self.decodable_chunk.exists():
            self.logger.register_log('segment_file NOTFOUND', self.decodable_chunk)
            error += ['segment_file NOTFOUND']
        if not self.reference_chunk.exists():
            self.logger.register_log('reference_segment NOTFOUND', self.reference_chunk)
            error += ['reference_segment NOTFOUND']
        msg = ', '.join(error)

        if msg:
            raise AbortError(msg)

    @property
    def decodable_chunk(self):
        return self.chunk_quality_paths.decodable_chunk

    @property
    def reference_chunk(self):
        return self.chunk_quality_paths.reference_chunk

    @property
    def chunk_quality_json(self):
        return self.chunk_quality_paths.chunk_quality_json
