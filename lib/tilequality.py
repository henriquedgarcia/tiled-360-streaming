from collections import defaultdict
from time import time

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.assets.worker import Worker
from lib.utils.context_utils import task, timer
from lib.utils.worker_utils import save_json, load_json, iter_frame, print_error


class TileChunkQuality(Worker, CtxInterface):
    quality_metrics: QualityMetrics
    chunk_quality_paths: ChunkQualityPaths

    def main(self):
        self.init()

        for _ in self.iterator():
            with task():
                self.work()

    def init(self):
        self.chunk_quality_paths = ChunkQualityPaths(self.ctx)
        self.quality_metrics = QualityMetrics(self.ctx)

    def iterator(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in reversed(self.quality_list):
                    for self.tiling in reversed(self.tiling_list):
                        for self.tile in self.tile_list:
                            yield

    def work(self):
        if self.skip_tile():
            raise AbortError('This chunk is OK.')

        self.assert_segments()

        with timer():
            reference_frames = iter_frame(self.reference_tile)
            tile_frame = iter_frame(self.tile_video)

            for frame in range(self.n_frames):
                print(f'\r\t{frame}', end='')

                chunk_frame = frame % 30
                if chunk_frame == 0:
                    self.chunk = 1 + frame // 30
                    chunk_quality = defaultdict(list)

                frame1 = next(reference_frames)
                frame2 = next(tile_frame)

                chunk_quality['ssim'].append(self.quality_metrics.ssim(frame1, frame2))
                chunk_quality['mse'].append(self.quality_metrics.mse(frame1, frame2))
                chunk_quality['s-mse'].append(self.quality_metrics.smse_nn(frame1, frame2))
                chunk_quality['ws-mse'].append(self.quality_metrics.wsmse(frame1, frame2))
                save_json(chunk_quality, self.chunk_quality_json)
            print(f'\n', end='')

    def skip_tile(self) -> bool:
        try:
            chunk_quality = load_json(self.chunk_quality_paths.chunk_quality_json)
        except FileNotFoundError:
            return False

        if len(chunk_quality['mse']) != int(self.config.gop):
            self.chunk_quality_paths.chunk_quality_json.unlink(missing_ok=True)
            self.logger.register_log(f'MISSING_FRAMES', self.chunk_quality_paths.chunk_quality_json)
            raise FileNotFoundError('Missing Frames')

        msg = []
        if 1 in chunk_quality['ssim']:
            self.logger.register_log(f'SSIM has 1.', self.segmenter_paths.chunk_video)
            msg.append('SSIM has 1.')

        if 0 in chunk_quality['mse']:
            self.logger.register_log('MSE has 0.', self.segmenter_paths.chunk_video)
            msg.append('MSE has 0.')

        if 0 in chunk_quality['ws-mse']:
            self.logger.register_log('WS-MSE has 0.', self.segmenter_paths.chunk_video)
            msg.append('WS-MSE has 0.')

        if 0 in chunk_quality['s-mse']:
            self.logger.register_log('S-MSE has 0.', self.segmenter_paths.chunk_video)
            msg.append('S-MSE has 0.')

        if len(msg) != 0:
            msg = "\n\t".join(msg)
            print_error(f'\t{msg}')

    def assert_segments(self):
        self.check_segment_file()
        self.check_reference_chunk()

    def check_segment_file(self):
        if not self.segmenter_paths.chunk_video.exists():
            self.logger.register_log('segment_file NOTFOUND', self.segmenter_paths.chunk_video)
            raise FileNotFoundError('segment_file NOTFOUND')

    def check_reference_chunk(self):
        if not self.chunk_quality_paths.reference_chunk.exists():
            self.logger.register_log('reference_segment NOTFOUND', self.chunk_quality_paths.reference_chunk)
            raise FileNotFoundError('reference_segment NOTFOUND')
