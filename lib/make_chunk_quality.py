import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import Config
from lib.assets.context import Context
from lib.assets.errors import AbortError
from lib.assets.paths.make_chunk_quality_paths import MakeChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.assets.worker import Worker
from lib.get_seen_tiles import GetTilesSeen
from lib.utils.context_utils import task
from lib.utils.util import iter_video


class MakeChunkQuality(Worker, MakeChunkQualityPaths):
    quality_metrics: QualityMetrics

    def main(self):
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self, verbose=True):
                self.work()

    def init(self):
        self.quality_metrics = QualityMetrics(self.ctx)

    def work(self):
        if self.chunk_quality_json_ok():
            raise AbortError('chunk_quality_json is ok')

        self.assert_chunk_and_reference()

        reference_frames = iter_video(self.reference_chunk)
        tile_frame = iter_video(self.decodable_chunk)
        error = []
        frame1 = frame2 = np.array([])
        chunk_quality = []

        print('')
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
                msg = ', '.join(error)
                self.logger.register_log(msg, self.decodable_chunk)
                raise AbortError(msg)

            ssim = self.quality_metrics.ssim(frame1, frame2)
            mse = self.quality_metrics.mse(frame1, frame2)
            s_mse = self.quality_metrics.smse_nn(frame1, frame2)
            ws_mse = self.quality_metrics.wsmse(frame1, frame2)

            chunk_quality.append((ssim, mse, s_mse, ws_mse))

        cols_name = ['ssim', 'mse', 's-mse', 'ws-mse']
        df = pd.DataFrame(chunk_quality, columns=cols_name)
        df.to_json(self.chunk_quality_json)

    def chunk_quality_json_ok(self) -> bool:
        def read_chunk_quality_json():
            filename = f'chunk{int(self.chunk)}_{self.config.rate_control}{self.ctx.quality}.json'
            old_name = self.chunk_quality_folder / filename
            if old_name.exists():
                old_name.rename(self.chunk_quality_json)
                chunk_quality = pd.read_json(self.chunk_quality_json)
                try:
                    del chunk_quality['frame']
                    chunk_quality.to_json(self.chunk_quality_json)
                except KeyError:
                    pass

            chunk_quality = pd.read_json(self.chunk_quality_json)
            if chunk_quality.size != int(self.gop) * 4:
                self.chunk_quality_json.unlink()
                self.logger.register_log(f'MISSING_FRAMES', self.chunk_quality_json)
                raise FileNotFoundError()

        try:
            read_chunk_quality_json()
            return True  # All OK
        except FileNotFoundError:
            return False

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


if __name__ == '__main__':
    os.chdir('../')

    # config_file = 'config_erp_qp.json'
    # config_file = 'config_cmp_crf.json'
    # config_file = 'config_erp_crf.json'
    # videos_file = 'videos_reversed.json'
    # videos_file = 'videos_lumine.json'
    # videos_file = 'videos_container0.json'
    # videos_file = 'videos_container1.json'
    # videos_file = 'videos_fortrek.json'
    # videos_file = 'videos_hp_elite.json'
    # videos_file = 'videos_alambique.json'
    # videos_file = 'videos_test.json'
    # videos_file = 'videos_full.json'

    # config_file = Path('config/config_cmp_qp.json')
    config_file = Path('config/config_erp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MakeChunkQuality(ctx)
