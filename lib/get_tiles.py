from collections import Counter
from time import time
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from py360tools import ERP, CMP, ProjectionBase
from py360tools.draw import draw

from lib.assets.autodict import AutoDict
from lib.assets.errors import GetTilesOkError, HMDDatasetError
from lib.assets.paths.gettilespaths import GetTilesPaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.worker import Worker
from lib.utils.worker_utils import (save_json, load_json, splitx, print_error,
                                    get_nested_value)


# "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
# respectively, to reorient during playback." - Author
# Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
# rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0

class GetTilesBase(Worker):
    hmd_dataset: dict
    projection_dict: dict['str', dict['str', ProjectionBase]]
    segmenter_paths: SegmenterPaths
    get_tiles_paths: GetTilesPaths
    user_hmd_data: dict

    def main(self):
        self.init()
        self.process()

    def init(self):
        self.get_tiles_paths = GetTilesPaths(self.ctx)
        self.segmenter_paths = SegmenterPaths(self.ctx)
        self.projection_dict = self.create_projections_dict()

    def for_each_user(self):
        ...

    def process(self):
        for self.ctx.name in self.ctx.name_list:
            for self.ctx.projection in self.ctx.projection_list:
                for self.ctx.tiling in self.ctx.tiling_list:
                    for self.ctx.user in self.ctx.users_list:
                        self.for_each_user()

    def create_projections_dict(self):
        projection_dict = AutoDict()
        for tiling in self.ctx.tiling_list:
            erp = build_projection(proj_name='erp', tiling=tiling,
                                   proj_res='1080x540', vp_res='660x540',
                                   fov_res=self.config.fov)
            cmp = build_projection(proj_name='cmp', tiling=tiling,
                                   proj_res='1080x540', vp_res='660x540',
                                   fov_res=self.config.fov)

            projection_dict['erp'][tiling] = erp
            projection_dict['cmp'][tiling] = cmp
        return projection_dict

    @property
    def user_hmd_data(self) -> list:
        return self.ctx.hmd_dataset[self.ctx.name + '_nas'][self.ctx.user]

    _results: dict

    @property
    def results(self):
        keys = [self.ctx.name, self.ctx.projection, self.ctx.tiling, self.ctx.user]
        try:
            value = get_nested_value(self._results, keys)
        except KeyError:
            value = None
        return value

    @results.setter
    def results(self, value):
        keys = [self.ctx.name, self.ctx.projection, self.ctx.tiling, self.ctx.user]
        get_nested_value(self._results, keys).update(value)

    def reset_results(self, data_type=dict):
        self._results = data_type()


class GetTilesReal(GetTilesBase):
    def for_each_user(self):
        print(f'==== GetTiles {self.ctx} ====')
        try:
            self.get_tiles_by_video()
        except (HMDDatasetError, GetTilesOkError) as e:
            print_error(f'\t{e.args[0]}')

    def get_tiles_by_video(self):
        self.check_get_tiles()
        self.assert_user_hmd_data()

        start = time()

        tiles_seen_by_frame = self.get_tiles_seen_by_frame(self.user_hmd_data)
        tiles_seen_by_chunk = self.get_tiles_seen_by_chunk(tiles_seen_by_frame)

        tiles_seen = {'frames': tiles_seen_by_frame,
                      'chunks': tiles_seen_by_chunk}

        save_json(tiles_seen, self.get_tiles_paths.user_tiles_seen_json)
        print(f'\ttime =  {time() - start}')

    def check_get_tiles(self):
        if not self.status.get_status('user_get_tiles_ok'):
            try:
                self.assert_user_tiles_seen_json()
            except FileNotFoundError:
                self.status.update_status('user_get_tiles_ok', False)
                return

            self.status.update_status('user_get_tiles_ok', True)
        raise GetTilesOkError('Get tiles is OK.')

    def assert_user_tiles_seen_json(self):
        size = self.get_tiles_paths.user_tiles_seen_json.stat().st_size

        if size == 0:
            self.get_tiles_paths.user_tiles_seen_json.unlink(missing_ok=True)
            raise FileNotFoundError

    def assert_user_hmd_data(self):
        if self.user_hmd_data == {}:
            self.logger.register_log(f'HMD samples is missing, '
                                     f'user{self.ctx.user}',
                                     self.get_tiles_paths.dataset_json)
            raise HMDDatasetError(f'HMD samples is missing, '
                                  f'user{self.ctx.user}')

    def get_tiles_seen_by_frame(self, user_hmd_data) -> list[list[str]]:
        if self.ctx.tiling == '1x1':
            return [["0"]] * self.config.n_frames

        tiles_seen_by_frame = []
        projection_obj = self.projection_dict[self.ctx.projection]
        projection_obj = projection_obj[self.ctx.tiling]

        for frame, yaw_pitch_roll in enumerate(user_hmd_data, 1):
            print(f'\r\tframe {frame:04d}/{self.config.n_frames}', end='')
            vptiles = projection_obj.get_vptiles(yaw_pitch_roll)
            vptiles: list[str] = list(map(str, map(int, vptiles)))
            tiles_seen_by_frame.append(vptiles)
        return tiles_seen_by_frame

    def get_tiles_seen_by_chunk(self, tiles_seen_by_frame):
        tiles_seen_by_chunk = {}

        if self.ctx.tiling == '1x1':
            duration = int(self.config.duration)
            return {str(i): ["0"] for i in range(1, duration + 1)}

        tiles_in_chunk = set()
        for frame, vptiles in enumerate(tiles_seen_by_frame):
            tiles_in_chunk.update(vptiles)

            if (frame + 1) % 30 == 0:
                chunk_id = frame // 30 + 1  # chunk start from 1
                tiles_seen_by_chunk[f'{chunk_id}'] = list(tiles_in_chunk)
                tiles_in_chunk.clear()
        return tiles_seen_by_chunk

    def count_tiles(self):
        if self.get_tiles_paths.counter_tiles_json.exists(): return

        self.results = load_json(self.get_tiles_paths.get_tiles_json)
        result = {}

        for self.ctx.tiling in self.ctx.tiling_list:
            if self.ctx.tiling == '1x1': continue

            # <editor-fold desc="Count tiles">
            tiles_counter_chunks = Counter()  # Collect tiling count

            for self.ctx.user in self.ctx.users_list:
                result_chunks = self.results[self.ctx.projection][self.ctx.name]
                result_chunks = result_chunks[self.ctx.tiling][self.ctx.user]
                result_chunks = result_chunks['chunks']

                for chunk in result_chunks:
                    tiles_counter_chunks = (tiles_counter_chunks
                                            + Counter(result_chunks[chunk]))
            # </editor-fold>

            print(tiles_counter_chunks)
            dict_tiles_counter_chunks = dict(tiles_counter_chunks)

            # <editor-fold desc="Normalize Counter">
            nb_chunks = sum(dict_tiles_counter_chunks.values())
            for self.ctx.tile in self.ctx.tile_list:
                try:
                    dict_tiles_counter_chunks[self.ctx.tile] /= nb_chunks
                except KeyError:
                    dict_tiles_counter_chunks[self.ctx.tile] = 0
            # </editor-fold>

            result[self.ctx.tiling] = dict_tiles_counter_chunks

        save_json(result, self.get_tiles_paths.counter_tiles_json)


class CreateJson(GetTilesBase):
    def process(self):
        for self.ctx.name in self.ctx.name_list:
            self.reset_results(AutoDict)
            for self.ctx.projection in self.ctx.projection_list:
                for self.ctx.tiling in self.ctx.tiling_list:
                    for self.ctx.user in self.ctx.users_list:
                        self.for_each_user()
            save_json(self.results, self.get_tiles_paths.get_tiles_json)

    def for_each_user(self):
        print(f'==== CreateJson {self.ctx} ====')
        tiles_seen = load_json(self.get_tiles_paths.user_tiles_seen_json)
        self.results.update(tiles_seen)


class HeatMap(GetTilesBase):
    def for_each_user(self):
        print(f'==== GetTiles {self.ctx} ====')
        try:
            self.heatmap()
        except (HMDDatasetError, GetTilesOkError) as e:
            print_error(f'\t{e.args[0]}')

    def heatmap(self):
        results = load_json(self.get_tiles_paths.counter_tiles_json)

        if self.ctx.tiling == '1x1': return

        filename = (f'heatmap_tiling_nasrabadi_28videos_{self.ctx.projection}_{self.ctx.name}_'
                    f'{self.ctx.tiling}_fov{self.config.fov}.png')
        heatmap_tiling = (self.get_tiles_paths.get_tiles_folder / filename)
        if heatmap_tiling.exists(): return

        tiling_result = results[self.ctx.tiling]

        h, w = splitx(self.ctx.tiling)[::-1]
        grade = np.zeros((h * w,))

        for item in tiling_result: grade[int(item)] = tiling_result[item]
        grade = grade.reshape((h, w))

        fig, ax = plt.subplots()
        im = ax.imshow(grade, cmap='jet', )
        ax.set_title(f'Tiling {self.ctx.tiling}')
        fig.colorbar(im, ax=ax, label='chunk frequency')

        # fig.show()
        fig.savefig(f'{heatmap_tiling}')


class TestGetTiles(GetTilesBase):
    def for_each_user(self):
        print(f'==== GetTiles {self.ctx} ====')
        try:
            self.plot_stats()
        except (HMDDatasetError, GetTilesOkError) as e:
            print_error(f'\t{e.args[0]}')

    def init(self):
        super().init()
        self.ctx.results = load_json(self.get_tiles_paths.get_tiles_json)
        pass

    def plot_stats(self):
        fig: plt.Figure
        ax: Union[np.ndarray, list]
        fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
        ax = list(ax.flat)
        ax: list[plt.Axes]

        for self.ctx.quality in '28':
            from collections import defaultdict
            result5 = defaultdict(list)  # By chunk

            for self.ctx.chunk in self.ctx.chunk_list:
                seen_tiles_metric = self.seen_tiles_metric[self.ctx.name][self.ctx.projection][self.ctx.tiling]
                [self.quality][self.ctx.user][self.ctx.chunk]
                tiles_list = seen_tiles_metric['time'].keys()

                result5[f'n_tiles'].append(len(tiles_list))
                for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
                    value = [seen_tiles_metric[self.metric][tile] for tile in tiles_list]
                    percentile = list(np.percentile(value, [0, 25, 50, 75, 100]))
                    # Tempo
                    # total
                    # de
                    # um
                    # chunk(sem
                    # decodificação
                    # paralela) (soma os tempos de decodificação
                    # dos tiles)
                    # result5[f'{self.metric}_sum'].append(np.sum(value))
                    # tempo
                    # médio
                    # de
                    # um
                    # chunk(com
                    # decodificação
                    # paralela) (média dos tempos de decodificação
                    # dos tiles)
                    result5[f'{self.metric}_avg'].append(np.average(value))
                    result5[f'{self.metric}_std'].append(np.std(value))
                    result5[f'{self.metric}_min'].append(percentile[0])
                    result5[f'{self.metric}_q1'].append(percentile[1])
                    result5[f'{self.metric}_median'].append(percentile[2])
                    result5[f'{self.metric}_q2'].append(percentile[3])
                    result5[f'{self.metric}_max'].append(percentile[4])

            ax[0].plot(result5['time_sum'], label=f'CRF{self.quality}')
            ax[1].plot(result5['time_avg'], label=f'CRF{self.quality}')
            ax[2].plot(result5['rate_sum'], label=f'CRF{self.quality}')
            ax[3].plot(result5['PSNR_avg'], label=f'CRF{self.quality}')
            ax[4].plot(result5['S-PSNR_avg'], label=f'CRF{self.quality}')
            ax[5].plot(result5['WS-PSNR_avg'], label=f'CRF{self.quality}')
            ax[6].plot(result5['n_tiles'], label=f'CRF{self.quality}')

            ax[0].set_title('time_sum')
            ax[1].set_title('time_avg')
            ax[2].set_title('rate_sum')
            ax[3].set_title('PSNR_avg')
            ax[4].set_title('S-PSNR_avg')
            ax[5].set_title('WS-PSNR_avg')
            ax[6].set_title('n_tiles')

        for a in ax[:-1]:
            a.legend(loc='upper right')

        fig.suptitle(f'{self.video} {self.ctx.tiling} - user {self.ctx.user}')
        fig.suptitle(f'{self.video} {self.ctx.tiling} - user {self.ctx.user}')
        fig.tight_layout()
        # fig.show()
        img_name = self.folder / f'{self.ctx.tiling}_user{self.ctx.user}.png'
        fig.savefig(img_name)
        plt.close(fig)


def build_projection(proj_name, proj_res, tiling, vp_res, fov_res) -> ProjectionBase:
    if proj_name == 'erp':
        projection = ERP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res)
    elif proj_name == 'cmp':
        projection = CMP(tiling=tiling, proj_res=proj_res, vp_res=vp_res, fov_res=fov_res)
    else:
        raise TypeError(f'Unknown projection name: {proj_name}')
    return projection


def print_tiles(proj: ProjectionBase, vptiles: list,
                yaw_pitch_roll: Union[tuple, np.ndarray] = None):
    if yaw_pitch_roll is not None:
        proj.yaw_pitch_roll = yaw_pitch_roll
    shape = proj.canvas.shape

    # convert fig_all_tiles_borders to RGB
    fig_all_tiles_borders = draw.draw_all_tiles_borders(projection=proj)
    fig_all_tiles_borders_ = np.array([fig_all_tiles_borders,
                                       fig_all_tiles_borders,
                                       fig_all_tiles_borders])
    fig_all_tiles_borders_ = fig_all_tiles_borders_.transpose([1, 2, 0])

    # Get vp tiles
    fig_vp_tiles = np.zeros(shape, dtype='uint8')
    for tile in vptiles:
        fig_vp_tiles = fig_vp_tiles + draw.draw_tile_border(projection=proj, idx=int(tile), lum=255)

    # get vp
    vp = draw.draw_vp_borders(projection=proj)

    # Compose
    fig_final = draw.compose(fig_all_tiles_borders_, fig_vp_tiles, (0, 255, 0))
    fig_final = draw.compose(fig_final, vp, (0, 0, 255))
    draw.show(fig_final)

# class TestGetTiles(GetTiles):
#     def init(self):
#         ctx.tiling_list.remove('1x1')
#         self.quality = '28'
#         self._get_tiles_data = {}
#         self.make_projections()
#
#     def main(self):
#         self.init()
#
#         for ctx.proj in [ctx.proj_list[1]]:
#             for ctx.name in ctx.name_list:
#                 for ctx.tiling in ctx.tiling_list:
#                     for ctx.user in [ctx.users_list[0]]:
#                         self.worker()
#
#     frame_n: int
#
#     def worker(self, overwrite=False):
#         print(f'{ctx.proj}, {ctx.name}, {ctx.tiling}, {ctx.user}')
#         yaw_pitch_roll_iter = iter(ctx.hmd_dataset[ctx.name+'_nas'][ctx.user])
#         ctx.frame_n = 0
#         for ctx.chunk in ctx.chunk_list:
#             for proj_frame in self.mount_chunk_frames():
#                 if self.output_video.exists():
#                     print(f' Exist. Skipping.')
#                     ctx.frame_n += 1
#                     next(yaw_pitch_roll_iter)
#                     continue
#
#                 ctx.projection_obj.yaw_pitch_roll = next(yaw_pitch_roll_iter)
#
#                 all_tiles_borders = ctx.projection_obj.draw_all_tiles_borders()
#                 vp_tiles = ctx.projection_obj.draw_vp_tiles()
#                 vp_mask = ctx.projection_obj.draw_vp_mask()
#                 try:
#                     vp_borders = ctx.projection_obj.draw_vp_borders()
#                 except IndexError:
#                     vp_borders = ctx.projection_obj.draw_vp_borders()
#
#                 viewport_array = ctx.projection_obj.get_vp_image(proj_frame)
#
#                 frame = compose(proj_frame_image=Image.fromarray(proj_frame),
#                                 all_tiles_borders_image=Image.fromarray(all_tiles_borders),
#                                 vp_tiles_image=Image.fromarray(vp_tiles),
#                                 vp_mask_image=Image.fromarray(vp_mask),
#                                 vp_borders_image=Image.fromarray(vp_borders),
#                                 vp_image=Image.fromarray(viewport_array),
#                                 )
#                 frame.save(self.output_video)
#                 ctx.frame_n += 1
#
#         print('')
#
#     def mount_chunk_frames(self, proj_shape=None, tile_shape=None,
#                            tiling_shape=None, chunk=None, seen_tiles=None
#                            ):
#         proj_h, proj_w = ctx.projection_obj.proj_shape
#         frame_proj = np.zeros((proj_h, proj_w, 3), dtype='uint8')
#         seen_tiles = self.get_user_samples()['chunks'][ctx.chunk]
#         tiles_reader = {ctx.tile: FFmpegReader(f'{paths.segment_video}').nextFrame()
#                         for ctx.tile in seen_tiles}
#         # seen_tiles = projection.get_vptiles()  # by frame
#
#         for frame in range(30):
#             for ctx.tile in seen_tiles:
#                 tile_h, tile_w = ctx.projection_obj.tile_shape
#                 tile_m, tile_n = idx2xy(int(ctx.tile), (ctx.projection_obj.tiling_h, ctx.projection_obj.tiling_w))
#                 tile_x, tile_y = tile_m * tile_w, tile_n * tile_h
#
#                 tile_frame = next(tiles_reader[ctx.tile])
#
#                 tile_resized = Image.fromarray(tile_frame).resize((tile_w, tile_h))
#                 tile_resized_array = np.asarray(tile_resized)
#                 frame_proj[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w, :] = tile_resized_array
#             yield frame_proj
#
#     @property
#     def output_video(self):
#         folder = paths.get_tiles_folder / 'videos' / f'{ctx.proj}_{ctx.name}_{ctx.tiling}'
#         folder.mkdir(parents=True, exist_ok=True)
#         output = folder / f"user{ctx.user}_{ctx.frame_n}.png"
#
#         return output


GetTiles = CreateJson
