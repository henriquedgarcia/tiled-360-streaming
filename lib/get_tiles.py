import pandas as pd
from collections import defaultdict
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from py360tools import ProjectionBase
from py360tools.draw import draw

from lib.assets.errors import GetTilesOkError, HMDDatasetError
from lib.make_tiles_seen import MakeTilesSeen
from lib.utils.util import print_error, load_json, splitx


# "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
# respectively, to reorient during playback." - Author
# Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
# rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0


class HeatMap(MakeTilesSeen):
    def for_each_user(self):
        print(f'==== GetTiles {self.ctx} ====')
        try:
            self.heatmap()
        except (HMDDatasetError, GetTilesOkError) as e:
            print_error(f'\t{e.args[0]}')

    def heatmap(self):
        results = load_json(self.counter_tiles_json)

        if self.tiling == '1x1': return

        filename = (f'heatmap_tiling_nasrabadi_28videos_{self.projection}_{self.name}_'
                    f'{self.tiling}_fov{self.config.fov}.png')
        heatmap_tiling = (self.seen_tiles_folder / filename)
        if heatmap_tiling.exists(): return

        tiling_result = results[self.tiling]

        h, w = splitx(self.tiling)[::-1]
        grade = np.zeros((h * w,))

        for item in tiling_result: grade[int(item)] = tiling_result[item]
        grade = grade.reshape((h, w))

        fig, ax = plt.subplots()
        im = ax.imshow(grade, cmap='jet', )
        ax.set_title(f'Tiling {self.tiling}')
        fig.colorbar(im, ax=ax, label='chunk frequency')

        # fig.show()
        fig.savefig(f'{heatmap_tiling}')


class TestMakeTilesSeen(MakeTilesSeen):
    def for_each_user(self):
        print(f'==== GetTiles {self.ctx} ====')
        try:
            self.plot_stats()
        except (HMDDatasetError, GetTilesOkError) as e:
            print_error(f'\t{e.args[0]}')

    def init(self):
        super().init()
        self.results = pd.read_pickle(self.seen_tiles_result_by_name)

    seen_tiles_metric: dict

    def plot_stats(self):
        fig: plt.Figure
        ax: Union[np.ndarray, list]
        fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
        ax = list(ax.flat)
        ax: list[plt.Axes]

        for self.quality in '28':
            result5 = defaultdict(list)  # By chunk
            self.seen_tiles_metric = {}
            for self.chunk in self.chunk_list:
                seen_tiles_metric = \
                    self.seen_tiles_metric[self.name][self.projection][self.tiling][self.quality][self.user][self.chunk]
                tiles_list = seen_tiles_metric['time'].keys()

                result5[f'n_tiles'].append(len(tiles_list))
                for self.metric in self.metric_list:
                    value = [seen_tiles_metric[self.metric][tile] for tile in tiles_list]
                    percentile = list(np.percentile(value, [0, 25, 50, 75, 100]))
                    # Tempo total de um chunk:
                    # sem decodificação paralela - soma os tempos de decodificação dos tiles
                    # com decodificação paralela - Usar apenas o maior tempo entre todos os tiles usados
                    result5[f'{self.metric}_sum'].append(np.sum(value))
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

        fig.suptitle(f'{self.name} {self.projection} {self.tiling} - user {self.user}')
        fig.tight_layout()
        # fig.show()
        img_name = self.seen_tiles_folder / f'{self.tiling}_user{self.user}.png'
        fig.savefig(img_name)
        plt.close(fig)


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

# class TestMakeTilesSeen(MakeTilesSeen):
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
