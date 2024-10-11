from collections import defaultdict

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.assets.worker import Worker
from lib.utils.context_utils import task, timer
from lib.utils.worker_utils import save_json, load_json, iter_frame


class TileChunkQuality(Worker, CtxInterface):
    quality_metrics: QualityMetrics
    tile_chunk_quality_paths: ChunkQualityPaths
    segmenter_paths: SegmenterPaths

    def init(self):
        self.tile_chunk_quality_paths = ChunkQualityPaths(self.ctx)
        self.quality_metrics = QualityMetrics(self.ctx)

    def main(self):
        self.init()

        for _ in self.iterator():
            with task():
                self.work()

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
                print(f'\t{frame}', end='')

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
            chunk_quality = load_json(self.tile_chunk_quality_paths.chunk_quality_json)
        except FileNotFoundError:
            return False

        if len(chunk_quality['mse']) != int(self.config.gop):
            self.chunk_quality_json.unlink(missing_ok=True)
            self.logger.register_log(f'MISSING_FRAMES', self.tile_chunk_quality_paths.chunk_quality_json)
            return False
        if len(chunk_quality['mse']) != int(self.config.gop):
            self.chunk_quality_json.unlink(missing_ok=True)
            self.logger.register_log(f'MISSING_FRAMES', self.tile_chunk_quality_paths.chunk_quality_json)
            return False

        return True

    def assert_segments(self):
        error = []
        if not self.chunk_video.exists():
            self.logger.register_log('segment_file NOTFOUND', self.chunk_video)
            error += ['segment_file NOTFOUND']
        if not self.reference_chunk.exists():
            self.logger.register_log('reference_segment NOTFOUND', self.reference_chunk)
            error += ['reference_segment NOTFOUND']
        msg = ', '.join(error)

        if msg:
            raise AbortError(msg)
        # All OK

    @property
    def reference_chunk(self):
        return self.tile_chunk_quality_paths.reference_chunk

    @property
    def reference_tile(self):
        return self.tile_chunk_quality_paths.reference_tile

    @property
    def tile_video(self):
        return self.tile_chunk_quality_paths.tile_video

    @property
    def chunk_video(self):
        return self.tile_chunk_quality_paths.chunk_video

    @property
    def chunk_quality_json(self):
        return self.tile_chunk_quality_paths.chunk_video
