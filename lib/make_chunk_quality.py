import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.assets.errors import AbortError
from lib.assets.paths.make_chunk_quality_paths import MakeChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import iter_video


class MakeChunkQuality(Worker, MakeChunkQualityPaths):
    quality_metrics: QualityMetrics

    def main(self):
        # self.check()

        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self):
                self.work()

    def init(self):
        self.quality_metrics = QualityMetrics(self.ctx)

    def work(self):
        if self.chunk_quality_pickle_ok():
            raise AbortError('chunk_quality_pickle is ok')

        reference_frames = iter_video(self.reference_chunk)
        tile_frame = iter_video(self.decodable_chunk)
        error = []
        frame1 = frame2 = np.array([])
        chunk_quality = []

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
        df.to_pickle(self.chunk_quality_pickle)

    def chunk_quality_pickle_ok(self) -> bool:
        try:
            chunk_quality = pd.read_pickle(self.chunk_quality_pickle)
            if len(chunk_quality['mse']) != int(self.gop):
                self.chunk_quality_json.unlink()
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
