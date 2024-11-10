import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from py360tools import ProjectionBase, ERP
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
from skvideo.io import FFmpegReader

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.tilequalitypaths import ChunkQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.utils.context_utils import context_quality, context_tile
from lib.utils.worker_utils import idx2xy, splitx, save_json, load_json


class ViewportPSNRProps(CtxInterface):
    dataset_data: dict
    erp_list: dict
    readers: dict
    seen_tiles: dict
    yaw_pitch_roll_frames: list
    video_frame_idx: int
    tile_h: float
    tile_w: float
    projection_obj: ProjectionBase

    quality_metrics: QualityMetrics
    tile_chunk_quality_paths: ChunkQualityPaths
    segmenter_paths: SegmenterPaths

    ## Methods #############################################
    def mount_frame(self, proj_frame, tiles_list, quality: str):
        readers = AutoDict()

        with context_quality(quality):
            for tile in tiles_list:
                with context_tile(tile):
                    readers[self.quality][self.tile] = cv.VideoCapture(f'{self.segmenter_paths.decodable_chunk}')

        for tile in tiles_list:
            is_ok, tile_frame = self.readers[quality][tile].read()

            tile_y, tile_x = self.projectionection_obj.tile_shape[-2] * n, self.projectionection_obj.tile_shape[-1] * m
            # tile_frame = cv.cvtColor(tile_frame, cv.COLOR_BGR2YUV)[:, :, 0]
            proj_frame[tile_y:tile_y + self.projectionection_obj.tile_shape[-2],
            tile_x:tile_x + self.projectionection_obj.tile_shape[-1], :] = tile_frame

    def tile_info(self, tile):
        info = {}
        m, n = idx2xy(idx=int(tile), shape=splitx(self.tiling)[::-1])
        info['nm'] = (n, m)
        tile_y, tile_x = self.tile_position_dict


    def output_exist(self, overwrite=False):
        if self.viewport_psnr_file.exists() and not overwrite:
            print(f'  The data file "{self.viewport_psnr_file}" exist.')
            return True
        return False

    @property
    def viewport_psnr_folder(self) -> Path:
        """
        Need None
        """
        folder = self.projectionect_path / f'ViewportPSNR'
        folder.mkdir(parents=True, exist_ok=True)
        return folder


class ViewportPSNR(ViewportPSNRProps):
    def init(self):
        self._get_tiles_data = {}
        self.make_projections(proj_res=("4320x2160", '3240x2160'), vp_shape=np.array([90, 110]) * 12)

    def main(self):
        self.init()
        for self.projection in self.projection_list:
            for self.name in self.name_list:
                for self.tiling in self.tiling_list:
                    for self.user in self.users_list:
                        for self.quality in self.quality_list:
                            self.worker()
                            # self.make_video()

    frame_n: int

    def worker(self):
        print(f'{self.projection}, {self.name}, {self.tiling}, {self.user}')

        if self.viewport_psnr_file.exists():
            print(f'\tThe file exist. Skipping')
            return

        self.frame_n = -1
        yaw_pitch_roll_iter = iter(self.dataset[self.name][self.user])
        seen_tiles = self.get_tiles_samples['chunks']

        proj_h, proj_w = self.projectionection_obj.proj_shape
        tile_h, tile_w = self.projectionection_obj.tile_shape
        qlt_by_frame = AutoDict()

        for self.chunk in self.chunk_list:
            print(f'Processing {self.name}_{self.tiling}_user{self.user}_chunk{self.chunk}')
            start = time.time()

            frame_proj = np.zeros((proj_h, proj_w, 3), dtype='uint8')
            frame_proj_ref = np.zeros((proj_h, proj_w, 3), dtype='uint8')

            tiles_reader_ref = {self.tile: FFmpegReader(f'{self.reference_segment}').nextFrame()
                                for self.tile in seen_tiles[self.chunk]}
            tiles_reader = {self.tile: FFmpegReader(f'{self.segment_video}').nextFrame()
                            for self.tile in seen_tiles[self.chunk]}

            for frame_idx in range(30):  # 30 frames per chunk
                self.frame_n += 1
                print(f'\tframe: {self.frame_n}. tiles {seen_tiles[self.chunk]}')

                yaw_pitch_roll = next(yaw_pitch_roll_iter)

                for self.tile in seen_tiles[self.chunk]:
                    tile_m, tile_n = idx2xy(int(self.tile), (self.N, self.M))
                    tile_x, tile_y = tile_m * tile_w, tile_n * tile_h

                    tile_frame_ref = next(tiles_reader_ref[self.tile])
                    tile_ref_resized = Image.fromarray(tile_frame_ref).resize((tile_w, tile_h))
                    tile_ref_resized_array = np.asarray(tile_ref_resized)
                    frame_proj_ref[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w, :] = tile_ref_resized_array
                    viewport_frame_ref = self.projectionection_obj.get_vp_image(frame_proj_ref,
                                                                                yaw_pitch_roll)  # .astype('float64')

                    tile_frame = next(tiles_reader[self.tile])
                    tile_resized = Image.fromarray(tile_frame).resize((tile_w, tile_h))
                    tile_resized_array = np.asarray(tile_resized)
                    frame_proj[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w, :] = tile_resized_array
                    viewport_frame = self.projectionection_obj.get_vp_image(frame_proj,
                                                                            yaw_pitch_roll)  # .astype('float64')

                    _mse = mse(viewport_frame_ref, viewport_frame)
                    _ssim = ssim(viewport_frame_ref, viewport_frame,
                                 data_range=255.0,
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)

                    try:
                        qlt_by_frame[self.projection][self.name][self.tiling][self.user][self.quality]['mse'].append(
                            _mse)
                        qlt_by_frame[self.projection][self.name][self.tiling][self.user][self.quality]['ssim'].append(
                            _ssim)
                    except AttributeError:
                        qlt_by_frame[self.projection][self.name][self.tiling][self.user][self.quality]['mse'] = [_mse]
                        qlt_by_frame[self.projection][self.name][self.tiling][self.user][self.quality]['ssim'] = [_ssim]

            print(f'\tchunk{self.chunk}_crf{self.quality}_frame{self.frame_n} - {time.time() - start: 0.3f} s')
        print('')
        save_json(qlt_by_frame, self.viewport_psnr_file)

    @property
    def yaw_pitch_roll_frames(self):
        return self.dataset[self.name][self.user]

    def make_video(self):
        if self.tiling == '1x1': return
        # vheight, vwidth  = np.array([90, 110]) * 6
        width, height = 576, 288

        # yaw_pitch_roll_frames = self.dataset[self.name][self.user]

        def debug_img() -> Path:
            folder = self.viewport_psnr_path / f'{self.projection}_{self.name}' / f"user{self.users_list[0]}_{self.tiling}"
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f"frame_{self.video_frame_idx}.jpg"

        for self.chunk in self.chunk_list:
            print(f'Processing {self.name}_{self.tiling}_user{self.user}_chunk{self.chunk}')
            seen_tiles = list(map(str, self.seen_tiles_by_chunks[self.chunk]))
            proj_frame = np.zeros((2160, 4320, 3), dtype='uint8')
            self.readers = AutoDict()
            start = time.time()

            # Operations by frame
            for chunk_frame_idx in range(int(self.gop)):  # 30 frames per chunk
                self.video_frame_idx = (int(self.chunk) - 1) * 30 + chunk_frame_idx

                if debug_img().exists():
                    print(f'Debug Video exist. State=[{self.video}][{self.tiling}][user{self.user}]')
                    return

                yaw_pitch_roll = self.yaw_pitch_roll_frames[self.video_frame_idx]

                # Build projection frame and get viewport
                self.mount_frame(proj_frame, seen_tiles, '0')
                # proj_frame = cv.cvtColor(proj_frame, cv.COLOR_BGR2RGB)[:, :, 0]
                # Image.fromarray(proj_frame[..., ::-1]).show()
                # Image.fromarray(cv.cvtColor(proj_frame, cv.COLOR_BGR2RGB)).show()

                viewport_frame = self.erp.get_vp_image(proj_frame, yaw_pitch_roll)  # .astype('float64')

                print(
                    f'\r    chunk{self.chunk}_crf{self.quality}_frame{chunk_frame_idx} - {time.time() - start: 0.3f} s',
                    end='')
                # vptiles = self.erp.get_vptiles(yaw_pitch_roll)

                # <editor-fold desc="Get and process frames">
                vp_frame_img = Image.fromarray(viewport_frame[..., ::-1])
                new_vwidth = int(np.round(height * vp_frame_img.width / vp_frame_img.height))
                vp_frame_img = vp_frame_img.resize((new_vwidth, height))

                proj_frame_img = Image.fromarray(proj_frame[..., ::-1])
                proj_frame_img = proj_frame_img.resize((width, height))
                # </editor-fold>

                cover_r = Image.new("RGB", (width, height), (255, 0, 0))
                cover_g = Image.new("RGB", (width, height), (0, 255, 0))
                cover_b = Image.new("RGB", (width, height), (0, 0, 255))
                cover_gray = Image.new("RGB", (width, height), (200, 200, 200))

                mask_all_tiles_borders = Image.fromarray(self.erp.draw_all_tiles_borders()).resize((width, height))
                mask_vp_tiles = Image.fromarray(self.erp.draw_vp_tiles(yaw_pitch_roll)).resize((width, height))
                mask_vp = Image.fromarray(self.erp.draw_vp_mask(lum=200)).resize((width, height))
                mask_vp_borders = Image.fromarray(self.erp.draw_vp_borders(yaw_pitch_roll)).resize((width, height))

                frame_img = Image.composite(cover_r, proj_frame_img, mask=mask_all_tiles_borders)
                frame_img = Image.composite(cover_g, frame_img, mask=mask_vp_tiles)
                frame_img = Image.composite(cover_gray, frame_img, mask=mask_vp)
                frame_img = Image.composite(cover_b, frame_img, mask=mask_vp_borders)

                img_final = Image.new('RGB', (width + new_vwidth + 2, height), (0, 0, 0))
                img_final.paste(frame_img, (0, 0))
                img_final.paste(vp_frame_img, (width + 2, 0))
                # img_final.show()
                img_final.save(debug_img())

            print('')


class ViewportPSNRGraphs(ViewportPSNRProps):
    _tiling: str
    _video: str
    _tile: str
    _user: str
    _quality: str
    dataset_data: dict
    dataset: dict
    erp_list: dict
    readers: AutoDict
    workfolder = None

    def main(self):
        # self.workfolder = super().workfolder / 'viewport_videos'  # todo: fix it
        self.workfolder.mkdir(parents=True, exist_ok=True)

        for self.video in self.video_list:
            self._get_tiles_data = load_json(self.get_tiles_json)
            self.erp_list = {tiling: ERP(tiling=tiling,
                                         proj_res=self.resolution,
                                         fov=self.fov)
                             for tiling in self.tiling_list}
            for self.tiling in self.tiling_list:
                self.erp = self.erp_list[self.tiling]
                self.tile_h, self.tile_w = self.erp.tile_shape[:2]
                for self.user in self.users_list:
                    self.yaw_pitch_roll_frames = self.dataset[self.name][self.user]
                    if self.output_exist(False): continue
                    # sse_frame = load_json(self.viewport_psnr_file)

                    for self.chunk in self.chunk_list:
                        for self.quality in self.quality_list:
                            for frame in range(int(self.fps)):  # 30 frames per chunk
                                self.worker()

    def worker(self, overwrite=False):
        pass


class CheckViewportPSNR(ViewportPSNR):
    def loop(self):
        self.workfolder.mkdir(parents=True, exist_ok=True)
        self.sse_frame: dict = {}
        self.frame: int = 0
        self.log = []
        debug1 = defaultdict(list)
        debug2 = defaultdict(list)
        # if self.output_exist(False): continue

        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    print(f'\r  Processing {self.vid_proj}_{self.name}_user{self.user}_{self.tiling}', end='')
                    viewport_psnr_file = self.projectionect_path / self.operation_folder / f'ViewportPSNR' / 'viewport_videos' / f'{self.vid_proj}_{self.name}' / f"user{self.user}_{self.tiling}.json"

                    try:
                        self.sse_frame = load_json(viewport_psnr_file)
                    except FileNotFoundError:
                        msg = f'FileNotFound: {self.viewport_psnr_file}'
                        debug1['video'].append(self.video)
                        debug1['tiling'].append(self.tiling)
                        debug1['user'].append(self.user)
                        debug1['msg'].append(msg)
                        continue

                    for self.quality in self.quality_list:
                        psnr = self.sse_frame[self.vid_proj][self.name][self.tiling][self.user][self.quality]['psnr']
                        n_frames = len(psnr)
                        more_than_100 = [x for x in psnr if x > 100]

                        if n_frames < (int(self.duration) * int(self.fps)):
                            msg = f'Few frames {n_frames}.'
                            debug2['video'].append(self.video)
                            debug2['tiling'].append(self.tiling)
                            debug2['user'].append(self.user)
                            debug2['quality'].append(self.quality)
                            debug2['error'].append('FrameError')
                            debug2['msg'].append(msg)

                        if len(more_than_100) > 0:
                            msg = f'{len(more_than_100)} values above PSNR 100 - max={max(psnr)}'
                            debug2['video'].append(self.video)
                            debug2['tiling'].append(self.tiling)
                            debug2['user'].append(self.user)
                            debug2['quality'].append(self.quality)
                            debug2['error'].append('ValueError')
                            debug2['msg'].append(msg)

        pd.DataFrame(debug1).to_csv("checkviewportpsnr1.csv", index=False)
        pd.DataFrame(debug2).to_csv("checkviewportpsnr2.csv", index=False)

        yield

    def worker(self, **kwargs):
        print(f'\rprocessing {self.vid_proj}_{self.name}_user{self.user}', end='')
