import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from py360tools import ERP, CMP
from py360tools import ProjectionBase
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from skvideo.io import FFmpegReader

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.gettilespaths import GetTilesPaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.userqualitypaths import UserQualityPaths
from lib.get_tiles import GetTiles
# import lib.erp as v360
from .assets.paths.tilequalitypaths import TileChunkQualityPaths
from .utils.worker_utils import save_json, load_json, splitx, idx2xy

pi = np.pi
pi2 = np.pi * 2


class UserProjectionMetricsProps(TileChunkQualityPaths, CtxInterface):
    seen_tiles_metric: AutoDict
    time_data: dict
    rate_data: dict
    qlt_data: dict
    get_tiles_data: dict

    @property
    def seen_metrics_folder(self) -> Path:
        folder = self.config.project_folder / 'UserProjectionMetrics'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def seen_metrics_json(self) -> Path:
        filename = f'seen_metrics_{self.config.dataset_file.stem}_{self.projection}_{self.name}.json'
        return self.seen_metrics_folder / filename

    def get_get_tiles(self):
        try:
            tiles_list = self.get_tiles_data[self.projection][self.name][self.tiling][self.user]['chunks'][self.chunk]
        except (KeyError, AttributeError):
            self.get_tiles_data = load_json(self.get_tiles_json, object_hook=dict)
            self.time_data = load_json(self.dectime_result_json, object_hook=dict)
            self.rate_data = load_json(self.bitrate_result_json, object_hook=dict)
            self.qlt_data = load_json(self.quality_result_json, object_hook=dict)
            tiles_list = self.get_tiles_data[self.projection][self.name][self.tiling][self.user]['chunks'][self.chunk]
        return tiles_list


class UserQuality(UserProjectionMetricsProps):
    tile_chunk_quality_paths: TileChunkQualityPaths
    dectime_paths: DectimePaths
    segmenter_paths: SegmenterPaths
    get_tiles_paths: GetTilesPaths
    user_quality_paths: UserQualityPaths
    seen_metrics_json: AutoDict
    
    def init(self):
        self.tile_chunk_quality_paths = TileChunkQualityPaths(self.ctx)
        self.dectime_paths = DectimePaths(self.ctx)
        self.segmenter_paths = SegmenterPaths(self.ctx)
        self.get_tiles_paths = GetTilesPaths(self.ctx)
        self.user_quality_paths = UserQualityPaths(self.ctx)

    def main(self):
        self.init()
        for _ in self.iterator():
            self.worker()

    def iterator(self):
        for self.name in self.name_list:
            self.seen_metrics_json = AutoDict()
            for self.projectionection in self.projectionection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.user in self.users_list:
                            for self.chunk in self.chunk_list:
                                yield
                                
            print(f'  Saving get tiles... ', end='')
            save_json(self.seen_tiles_metric, self.seen_metrics_json)
            print(f'  Finished.')

        self.graphs1()
        self.graphs2()
        print('')

    def worker(self):
        for self.tile in self.get_get_tiles():
            dectime_val = self.time_data[self.projection][self.name][self.tiling][self.quality][self.tile][self.chunk]
            bitrate_val = self.rate_data[self.projection][self.name][self.tiling][self.quality][self.tile][self.chunk]
            quality_val = self.qlt_data[self.projection][self.name][self.tiling][self.quality][self.tile][self.chunk]

            try:
                metrics_result = \
                    self.seen_tiles_metric[self.projection][self.name][self.tiling][self.quality][self.user][self.chunk][
                        self.tile]
            except (NameError, AttributeError, KeyError):
                self.seen_tiles_metric = AutoDict()
                metrics_result = \
                    self.seen_tiles_metric[self.projection][self.name][self.tiling][self.quality][self.user][self.chunk][
                        self.tile]

            metrics_result['time'] = float(np.average(dectime_val))
            metrics_result['rate'] = float(bitrate_val)
            metrics_result['time_std'] = float(np.std(dectime_val))
            metrics_result['PSNR'] = quality_val['PSNR']
            metrics_result['WS-PSNR'] = quality_val['WS-PSNR']
            metrics_result['S-PSNR'] = quality_val['S-PSNR']

    def graphs1(self):
        # for each user plot quality in function of chunks
        def img_name():
            folder = self.seen_metrics_folder / f'1_{self.name}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f'{self.tiling}_user{self.user}.png'

        def loop_video_tiling_user():
            for self.video in self.video_list:
                # for self.tiling in ['6x4']:
                for self.tiling in self.tiling_list:
                    for self.user in self.users_list:
                        yield

        for _ in loop_video_tiling_user():
            if img_name().exists(): continue
            print(f'\r{img_name()}', end='')

            fig: plt.Figure
            ax: list[plt.Axes]
            fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
            ax: plt.Axes
            ax = np.ravel(ax)
            result_by_quality = AutoDict()  # By quality by chunk

            for self.quality in self.quality_list:
                for self.chunk in self.chunk_list:
                    # <editor-fold desc="get seen_tiles_metric">
                    try:
                        seen_tiles_metric = \
                            self.seen_tiles_metric[self.projection][self.name][self.tiling][self.quality][self.user][
                                self.chunk]
                    except (KeyError, AttributeError):
                        self.seen_tiles_metric = load_json(self.seen_metrics_json)
                        seen_tiles_metric = \
                            self.seen_tiles_metric[self.projection][self.name][self.tiling][self.quality][self.user][
                                self.chunk]
                    # </editor-fold>

                    tiles_list = seen_tiles_metric['time'].keys()
                    try:
                        result_by_quality[self.quality][f'n_tiles'].append(len(tiles_list))
                    except AttributeError:
                        result_by_quality[self.quality][f'n_tiles'] = [len(tiles_list)]

                    for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
                        tile_metric_value = [seen_tiles_metric[self.metric][tile] for tile in tiles_list]
                        percentile = list(np.percentile(tile_metric_value, [0, 25, 50, 75, 100]))
                        try:
                            result_by_quality[self.quality][f'{self.metric}_sum'].append(
                                np.sum(
                                    tile_metric_value))  # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)
                        except AttributeError:
                            result_by_quality[self.quality] = defaultdict(list)
                            result_by_quality[self.quality][f'{self.metric}_sum'].append(
                                np.sum(
                                    tile_metric_value))  # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)

                        result_by_quality[self.quality][f'{self.metric}_avg'].append(
                            np.average(
                                tile_metric_value))  # tempo médio de um chunk (com decodificação paralela) (média dos tempos de decodificação dos tiles)
                        result_by_quality[self.quality][f'{self.metric}_std'].append(np.std(tile_metric_value))
                        result_by_quality[self.quality][f'{self.metric}_min'].append(percentile[0])
                        result_by_quality[self.quality][f'{self.metric}_q1'].append(percentile[1])
                        result_by_quality[self.quality][f'{self.metric}_median'].append(percentile[2])
                        result_by_quality[self.quality][f'{self.metric}_q2'].append(percentile[3])
                        result_by_quality[self.quality][f'{self.metric}_max'].append(percentile[4])

                ax[0].plot(result_by_quality[self.quality]['time_sum'], label=f'CRF{self.quality}')
                ax[1].plot(result_by_quality[self.quality]['time_avg'], label=f'CRF{self.quality}')
                ax[2].plot(result_by_quality[self.quality]['rate_sum'], label=f'CRF{self.quality}')
                ax[3].plot(result_by_quality[self.quality]['PSNR_avg'], label=f'CRF{self.quality}')
                ax[4].plot(result_by_quality[self.quality]['S-PSNR_avg'], label=f'CRF{self.quality}')
                ax[5].plot(result_by_quality[self.quality]['WS-PSNR_avg'], label=f'CRF{self.quality}')
                ax[6].plot(result_by_quality[self.quality]['n_tiles'], label=f'CRF{self.quality}')

            ax[0].set_title('Tempo de decodificação total')
            ax[1].set_title('Tempo médio de decodificação')
            ax[2].set_title('Taxa de bits total')
            ax[3].set_title(f'PSNR médio')
            ax[4].set_title('S-PSNR médio')
            ax[5].set_title('WS-PSNR médio')
            ax[6].set_title('Número de ladrilhos')
            for a in ax[:-2]: a.legend(loc='upper right')

            name = self.name.replace('_nas', '').replace('_', ' ').title()
            fig.suptitle(f'{name} {self.tiling} - user {self.user}')
            fig.tight_layout()
            fig.show()
            fig.savefig(img_name())
            plt.close(fig)

    def graphs2(self):
        def img_name():
            folder = self.seen_metrics_folder / f'2_aggregate'
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f'{self.name}_{self.tiling}.png'

        def loop_video_tiling():
            for self.video in self.video_list:
                self.seen_tiles_metric = load_json(self.seen_metrics_json)
                for self.tiling in self.tiling_list:
                    yield

        # Compara usuários
        for _ in loop_video_tiling():
            if img_name().exists(): continue
            print(img_name(), end='')

            fig: plt.Figure
            ax: list[plt.Axes]
            fig, ax = plt.subplots(2, 5, figsize=(15, 5), dpi=200)
            ax: plt.Axes
            ax = np.ravel(ax)

            for self.quality in self.quality_list:
                result_lv2 = defaultdict(list)  # By chunk

                for self.user in self.users_list:
                    result_lv1 = defaultdict(list)  # By chunk

                    for self.chunk in self.chunk_list:
                        seen_tiles_data = \
                            self.seen_tiles_metric[self.projection][self.name][self.tiling][self.quality][self.user][
                                self.chunk]
                        tiles_list = seen_tiles_data['time'].keys()

                        result_lv1[f'n_tiles'].append(len(tiles_list))
                        for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            value = [seen_tiles_data[self.metric][tile] for tile in tiles_list]
                            percentile = list(np.percentile(value, [0, 25, 50, 75, 100]))
                            result_lv1[f'{self.metric}_sum'].append(
                                np.sum(
                                    value))  # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)
                            result_lv1[f'{self.metric}_avg'].append(
                                np.average(
                                    value))  # tempo médio de um chunk (com decodificação paralela) (média dos tempos de decodificação dos tiles)
                            result_lv1[f'{self.metric}_std'].append(np.std(value))
                            result_lv1[f'{self.metric}_min'].append(percentile[0])
                            result_lv1[f'{self.metric}_q1'].append(percentile[1])
                            result_lv1[f'{self.metric}_median'].append(percentile[2])
                            result_lv1[f'{self.metric}_q2'].append(percentile[3])
                            result_lv1[f'{self.metric}_max'].append(percentile[4])

                    # each metrics represent the metrics by complete reproduction of the one vídeo with one tiling in one quality for one user
                    result_lv2[f'time_total'].append(
                        np.sum(result_lv1[f'time_sum']))  # tempo total sem decodificação paralela
                    result_lv2[f'time_avg_sum'].append(
                        np.average(result_lv1[f'time_sum']))  # tempo médio sem decodificação paralela
                    result_lv2[f'time_total_avg'].append(
                        np.sum(result_lv1[f'time_avg']))  # tempo total com decodificação paralela
                    result_lv2[f'time_avg_avg'].append(
                        np.average(result_lv1[f'time_avg']))  # tempo total com decodificação paralela
                    result_lv2[f'rate_total'].append(np.sum(result_lv1[f'rate_sum']))  # taxa de bits sempre soma
                    result_lv2[f'psnr_avg'].append(np.average(result_lv1[f'PSNR_avg']))  # qualidade sempre é média
                    result_lv2[f'ws_psnr_avg'].append(np.average(result_lv1[f'WS-PSNR_avg']))
                    result_lv2[f's_psnr_avg'].append(np.average(result_lv1[f'S-PSNR_avg']))
                    result_lv2[f'n_tiles_avg'].append(np.average(result_lv1[f'n_tiles']))
                    result_lv2[f'n_tiles_total'].append(np.sum(result_lv1[f'n_tiles']))

                result4_df = pd.DataFrame(result_lv2)
                # result4_df = result4_df.sort_values(by=['rate_total'])
                x = list(range(len(result4_df['time_total'])))
                ax[0].bar(x, result4_df['time_total'], label=f'CRF{self.quality}')
                ax[1].bar(x, result4_df['time_avg_sum'], label=f'CRF{self.quality}')
                ax[2].bar(x, result4_df['time_total_avg'], label=f'CRF{self.quality}')
                ax[3].bar(x, result4_df['time_avg_avg'], label=f'CRF{self.quality}')
                ax[4].bar(x, result4_df['rate_total'], label=f'CRF{self.quality}')
                ax[5].bar(x, result4_df['psnr_avg'], label=f'CRF{self.quality}')
                ax[6].bar(x, result4_df['ws_psnr_avg'], label=f'CRF{self.quality}')
                ax[7].bar(x, result4_df['s_psnr_avg'], label=f'CRF{self.quality}')
                ax[8].bar(x, result4_df['n_tiles_avg'], label=f'CRF{self.quality}')
                ax[9].bar(x, result4_df['n_tiles_total'], label=f'CRF{self.quality}')

                ax[0].set_title('time_total')
                ax[1].set_title('time_avg_sum')
                ax[2].set_title('time_total_avg')
                ax[3].set_title('time_avg_avg')
                ax[4].set_title('rate_total')
                ax[5].set_title('psnr_avg')
                ax[6].set_title('ws_psnr_avg')
                ax[7].set_title('s_psnr_avg')
                ax[8].set_title('n_tiles_avg')
                ax[9].set_title('n_tiles_total')

            for a in ax[:-2]:
                a.legend(loc='upper right')

            fig.suptitle(f'{self.video} {self.tiling}')
            fig.tight_layout()
            # fig.show()
            fig.savefig(img_name)
            img_name = img_name().parent / f'{self.tiling}_{self.name}.png'
            fig.savefig(img_name)
            plt.close(fig)

            # result3[f'time_avg_total'].append(np.average(result4[f'time_total']))  # comparando entre usuários usamos o tempo médio
            # result3[f'time_avg_avg_sum'].append(np.sum(result4[f'time_avg_sum']))  # tempo médio sem paralelismo
            # result3[f'time_avg_avg'].append(np.average(result4[f'time_avg']))  # tempo total com decodificação paralela
            # result3[f'rate_total'].append(np.sum(result4[f'rate_sum']))  # taxa de bits sempre soma
            # result3[f'psnr_avg'].append(np.average(result4[f'PSNR_avg']))  # qualidade sempre é média
            # result3[f'ws_psnr_avg'].append(np.average(result4[f'WS-PSNR']))
            # result3[f's_psnr_avg'].append(np.average(result4[f'S-PSNR']))


class ViewportPSNRProps(GetTiles):
    _tiling: str
    _video: str
    _tile: str
    _user: str
    _erp_list: dict[str, ERP]
    _seen_tiles: dict
    _erp: ERP
    dataset_data: dict
    erp_list: dict
    readers: dict
    seen_tiles: dict
    yaw_pitch_roll_frames: list
    video_frame_idx: int
    tile_h: float
    tile_w: float
    projection_obj: ProjectionBase

    ## Lists #############################################
    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list

    def make_projections(self, proj_res=('720x360', '540x360'), vp_res='440x294'):
        projection_dict = AutoDict()

        for self.tiling in self.tiling_list:
            erp = ERP(tiling=self.tiling,
                      proj_res=proj_res[0],
                      vp_res=vp_res,
                      fov_res=self.fov)
            cmp = CMP(tiling=self.tiling,
                      proj_res=proj_res[1],
                      vp_res=vp_res,
                      fov=self.fov)
            projection_dict['erp'][self.tiling] = erp
            projection_dict['cmp'][self.tiling] = cmp
            return projection_dict

    @property
    def projection_obj(self) -> ProjectionBase:
        return self.projectionection_dict[self.projection][self.tiling]

    ## Properties #############################################
    #
    # @property
    # def erp(self) -> v360.ERP:
    #     """
    #     self.erp_list[self.tiling]
    #     :return:
    #     """
    #     self._erp = self.erp_list[self.tiling]
    #     return self._erp
    #
    # @erp.setter
    # def erp(self, value: v360.ERP):
    #     self._erp = value
    #
    # @property
    # def erp_list(self) -> dict:
    #     """
    #     {tiling: vp.ERP(tiling, self.resolution, self.fov, vp_shape=np.array([90, 110]) * 6) for tiling in self.tiling_list}
    #     :return:
    #     """
    #     while True:
    #         try:
    #             return self._erp_list
    #         except AttributeError:
    #             print(f'Loading list of ERPs')
    #             self._erp_list = {tiling: ERP(tiling=tiling, proj_res=self.resolution,
    #                                           fov=self.fov, vp_shape=np.array([90, 110]) * 6)
    #                               for tiling in self.tiling_list}

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    ## Paths #############################################
    @property
    def viewport_psnr_path(self) -> Path:
        folder = self.projectionect_path / 'ViewportPSNR'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def viewport_psnr_file(self) -> Path:
        folder = self.viewport_psnr_path / f'{self.projection}_{self.name}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"user{self.user}_{self.tiling}.json"

    ## Methods #############################################
    def mount_frame(self, proj_frame, tiles_list, quality):
        self.quality = quality
        for self.tile in tiles_list:
            try:
                is_ok, tile_frame = self.readers[self.quality][self.tile].read()
            except (AttributeError, KeyError):
                self.readers = {self.quality: {self.tile: cv.VideoCapture(f'{self.segment_video}')}}
                is_ok, tile_frame = self.readers[self.quality][self.tile].read()

            m, n = idx2xy(idx=int(self.tile), shape=splitx(self.tiling)[::-1])
            tile_y, tile_x = self.projectionection_obj.tile_shape[-2] * n, self.projectionection_obj.tile_shape[-1] * m
            # tile_frame = cv.cvtColor(tile_frame, cv.COLOR_BGR2YUV)[:, :, 0]
            proj_frame[tile_y:tile_y + self.projectionection_obj.tile_shape[-2],
            tile_x:tile_x + self.projectionection_obj.tile_shape[-1], :] = tile_frame

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
                    viewport_frame = self.projectionection_obj.get_vp_image(frame_proj, yaw_pitch_roll)  # .astype('float64')

                    _mse = mse(viewport_frame_ref, viewport_frame)
                    _ssim = ssim(viewport_frame_ref, viewport_frame,
                                 data_range=255.0,
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)

                    try:
                        qlt_by_frame[self.projection][self.name][self.tiling][self.user][self.quality]['mse'].append(_mse)
                        qlt_by_frame[self.projection][self.name][self.tiling][self.user][self.quality]['ssim'].append(_ssim)
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


# class CheckViewportPSNR(ViewportPSNR):
#
#     @property
#     def quality_list(self) -> list[str]:
#         quality_list: list = self.config['quality_list']
#         try:
#             quality_list.remove('0')
#         except ValueError:
#             pass
#         return quality_list
#
#     def loop(self):
#
#         self.workfolder.mkdir(parents=True, exist_ok=True)
#         self.sse_frame: dict = {}
#         self.frame: int = 0
#         self.log = []
#         debug1 = defaultdict(list)
#         debug2 = defaultdict(list)
#         # if self.output_exist(False): continue
#
#         for self.video in self.videos_list:
#             for self.tiling in self.tiling_list:
#                 for self.user in self.users_list:
#                     print(f'\r  Processing {self.vid_proj}_{self.name}_user{self.user}_{self.tiling}', end='')
#                     viewport_psnr_file = self.projectionect_path / self.operation_folder / f'ViewportPSNR' / 'viewport_videos' / f'{self.vid_proj}_{self.name}' / f"user{self.user}_{self.tiling}.json"
#
#                     try:
#                         self.sse_frame = load_json(viewport_psnr_file)
#                     except FileNotFoundError:
#                         msg = f'FileNotFound: {self.viewport_psnr_file}'
#                         debug1['video'].append(self.video)
#                         debug1['tiling'].append(self.tiling)
#                         debug1['user'].append(self.user)
#                         debug1['msg'].append(msg)
#                         continue
#
#                     for self.quality in self.quality_list:
#                         psnr = self.sse_frame[self.vid_proj][self.name][self.tiling][self.user][self.quality]['psnr']
#                         n_frames = len(psnr)
#                         more_than_100 = [x for x in psnr if x > 100]
#
#                         if n_frames < (int(self.duration) * int(self.fps)):
#                             msg = f'Few frames {n_frames}.'
#                             debug2['video'].append(self.video)
#                             debug2['tiling'].append(self.tiling)
#                             debug2['user'].append(self.user)
#                             debug2['quality'].append(self.quality)
#                             debug2['error'].append('FrameError')
#                             debug2['msg'].append(msg)
#
#                         if len(more_than_100) > 0:
#                             msg = f'{len(more_than_100)} values above PSNR 100 - max={max(psnr)}'
#                             debug2['video'].append(self.video)
#                             debug2['tiling'].append(self.tiling)
#                             debug2['user'].append(self.user)
#                             debug2['quality'].append(self.quality)
#                             debug2['error'].append('ValueError')
#                             debug2['msg'].append(msg)
#
#         pd.DataFrame(debug1).to_csv("checkviewportpsnr1.csv", index=False)
#         pd.DataFrame(debug2).to_csv("checkviewportpsnr2.csv", index=False)
#
#         yield
#
#     def worker(self, **kwargs):
#         print(f'\rprocessing {self.vid_proj}_{self.name}_user{self.user}', end='')


def show(array):
    Image.fromarray(array).show()


UserMetricsOptions = {
    '4': UserQuality,  # 4
    '5': ViewportPSNR,  # 5
    }