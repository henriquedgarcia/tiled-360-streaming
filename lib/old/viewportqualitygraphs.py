import time
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image
from py360tools import ERP, ProjectionBase, Viewport
from py360tools.draw import draw

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.viewportqualitypaths import ViewportQualityPaths
from lib.assets.qualitymetrics import QualityMetrics
from lib.utils.util import build_projection, load_json, splitx, idx2xy
from lib.utils.context_utils import context_quality, context_tile


class ViewportQualityProps(CtxInterface):
    dataset_data: dict
    erp_list: dict
    readers: dict
    seen_tiles: dict
    yaw_pitch_roll_frames: list
    video_frame_idx: int
    tile_h: float
    tile_w: float
    projection_obj: Viewport

    quality_metrics: QualityMetrics
    viewport_quality_paths: ViewportQualityPaths

    def mount_frame(self, proj_frame, tiles_list, quality: str):
        readers = AutoDict()

        with context_quality(quality):
            for tile in tiles_list:
                with context_tile(tile):
                    readers[self.quality][self.tile] = cv.VideoCapture(f'{self.viewport_quality_paths.decodable_chunk}')

        self.projection_obj = build_projection(proj_name=self.ctx.projection,
                                               tiling=self.ctx.tiling,
                                               proj_res=self.ctx.scale,
                                               vp_res='1320x1080',
                                               fov_res=self.ctx.fov)

        if proj_name == 'erp':
            projection = ERP(tiling=tiling, proj_res=proj_res)
        elif proj_name == 'cmp':
            projection = CMP(tiling=tiling, proj_res=proj_res)
        else:
            raise TypeError(f'Unknown projection name: {proj_name}')
        return Viewport('800x800', '90x90', projection)

        tile_N, tile_M = self.projection_obj.tiling.shape
        for tile in tiles_list:
            tile_m, tile_n = idx2xy(int(tile), (tile_N, tile_M))
            is_ok, tile_frame = self.readers[quality][tile].read()

            tile_y, tile_x = self.projection_obj.tiling.tile_shape[-2] * tile_n, self.projection_obj.viewport.tile_shape[-1] * tile_m
            # tile_frame = cv.cvtColor(tile_frame, cv.COLOR_BGR2YUV)[:, :, 0]
            (proj_frame[tile_y:tile_y + self.projection_obj.tiling.tile_shape[-2],
             tile_x:tile_x + self.projection_obj.tiling.tile_shape[-1], :]) = tile_frame

    def make_video(self):
        if self.tiling == '1x1': return
        # vheight, vwidth  = np.array([90, 110]) * 6
        width, height = 576, 288

        # yaw_pitch_roll_frames = self.dataset[self.name][self.user]

        def debug_img() -> Path:
            folder = self.viewport_quality_paths.viewport_quality_folder / f'{self.projection}_{self.name}' / f"user{self.users_list_by_name[0]}_{self.tiling}"
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
                    print(f'Debug Video exist. State=[{self.name}][{self.tiling}][user{self.user}]')
                    return

                yaw_pitch_roll = self.yaw_pitch_roll_frames[self.video_frame_idx]

                # Build projection frame and get viewport
                self.mount_frame(proj_frame, seen_tiles, '0')
                # proj_frame = cv.cvtColor(proj_frame, cv.COLOR_BGR2RGB)[:, :, 0]
                # Image.fromarray(proj_frame[..., ::-1]).show()
                # Image.fromarray(cv.cvtColor(proj_frame, cv.COLOR_BGR2RGB)).show()

                viewport_frame = self.projection_obj.extract_viewport(proj_frame, yaw_pitch_roll)  # .astype('float64')

                print(f'\r    chunk{self.chunk}_crf{self.quality}_frame{chunk_frame_idx} '
                      f'- {time.time() - start: 0.3f} s', end='')
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
                mask_all_tiles_borders = Image.fromarray(draw.draw_all_tiles_borders()).resize((width, height))
                mask_vp_tiles = Image.fromarray(draw.draw_vp_tiles(projection=self.projection_obj)).resize((width, height))
                mask_vp = Image.fromarray(draw.draw_vp_mask(projection=self.projection_obj, lum=200)).resize((width, height))
                mask_vp_borders = Image.fromarray(draw.draw_vp_borders(projection=self.projection_obj)).resize((width, height))

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

    def tile_info(self, tile):
        info = {}
        m, n = idx2xy(idx=int(tile), shape=splitx(self.tiling)[::-1])
        info['nm'] = (n, m)
        tile_y, tile_x = self.tile_position_dict

    def output_exist(self, overwrite=False):
        if self.viewport_quality_paths.user_viewport_quality_json.exists() and not overwrite:
            print(f'  The data file "{self.viewport_quality_paths.user_viewport_quality_json}" exist.')
            return True
        return False


class ViewportQualityGraphs(ViewportQualityProps):
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

        for self.name in self.name_list:
            self._get_tiles_data = load_json(self.get_tiles_json)
            self.erp_list = {}
            for tiling in self.tiling_list:
                self.erp_list = {tiling: ERP(tiling=tiling,
                                             proj_res=self.resolution,
                                             )
                                 for tiling in self.tiling_list}
            for self.tiling in self.tiling_list:
                self.projection_obj = self.erp_list[self.tiling]
                self.tile_h, self.tile_w = self.erp.tile_shape[:2]
                for self.user in self.users_list_by_name:
                    self.yaw_pitch_roll_frames = self.dataset[self.name][self.user]
                    if self.output_exist(False): continue
                    # sse_frame = load_json(self.viewport_psnr_file)

                    for self.chunk in self.chunk_list:
                        for self.quality in self.quality_list:
                            for self.frame in range(int(self.fps)):  # 30 frames per chunk
                                self.worker()

    def worker(self, overwrite=False):
        pass
