from collections import defaultdict
from time import time

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.assets.worker import Worker
from lib.utils.worker_utils import save_json, load_json, iter_frame, print_error


class TileChunkQuality(Worker, CtxInterface):
    quality_metrics: QualityMetrics
    tile_chunk_quality_paths: ChunkQualityPaths
    segmenter_paths: SegmenterPaths

    def main(self):
        self.init()

        for _ in self.iterator():
            try:
                print(f'==== TileChunkQuality {self.ctx} ====')
                self.work()
            except (AbortError, FileNotFoundError) as e:
                print_error('\t' + e.args[0])
            except ValueError as e:
                self.logger.register_log('Cant decode Chunk.', self.segmenter_paths.chunk_video)
                print_error('\t' + e.args[0])
                try:
                    self.segmenter_paths.chunk_video.unlink()
                except PermissionError:
                    print_error('\tCant remove chunk_video.')
                    self.logger.register_log('Cant remove chunk_video.', self.segmenter_paths.chunk_video)

    def init(self):
        self.tile_chunk_quality_paths = ChunkQualityPaths(self.ctx)
        self.segmenter_paths = SegmenterPaths(self.ctx)
        self.quality_metrics = QualityMetrics(self.ctx)

    def iterator(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in reversed(self.quality_list):
                    for self.tiling in reversed(self.tiling_list):
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                yield

    def work(self):
        self.check_tile_chunk_quality()
        self.assert_segments()

        start = time()

        chunk_quality = defaultdict(list)

        iter_reference_segment = iter_frame(self.tile_chunk_quality_paths.reference_chunk)
        iter_segment = iter_frame(self.segmenter_paths.chunk_video)
        zip_frames = zip(iter_reference_segment, iter_segment)

        for frame, (frame1, frame2) in enumerate(zip_frames):
            print(f'\r\t{frame=}', end='')
            chunk_quality['ssim'].append(self.quality_metrics.ssim(frame1, frame2))
            chunk_quality['mse'].append(self.quality_metrics.mse(frame1, frame2))
            chunk_quality['s-mse'].append(self.quality_metrics.smse_nn(frame1, frame2))
            chunk_quality['ws-mse'].append(self.quality_metrics.wsmse(frame1, frame2))

        save_json(chunk_quality, self.tile_chunk_quality_paths.chunk_quality_json)
        print(f"\ttime={time() - start}.")

    def check_tile_chunk_quality(self):
        if not self.status.get_status('tile_chunk_quality_json_ok'):
            try:
                self.assert_tile_chunk_quality_json()
            except FileNotFoundError:
                self.status.update_status('tile_chunk_quality_json_ok', False)
                return
            self.status.update_status('tile_chunk_quality_json_ok', True)
        raise AbortError('tile_chunk_quality_json is OK.')

    def assert_tile_chunk_quality_json(self):
        chunk_quality = load_json(self.tile_chunk_quality_paths.chunk_quality_json)

        if len(chunk_quality['mse']) != int(self.config.gop):
            self.tile_chunk_quality_paths.chunk_quality_json.unlink(missing_ok=True)
            self.logger.register_log(f'MISSING_FRAMES', self.tile_chunk_quality_paths.chunk_quality_json)
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
        if not self.tile_chunk_quality_paths.reference_chunk.exists():
            self.logger.register_log('reference_segment NOTFOUND', self.tile_chunk_quality_paths.reference_chunk)
            raise FileNotFoundError('reference_segment NOTFOUND')
