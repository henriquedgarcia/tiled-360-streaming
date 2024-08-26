from lib.assets.worker import Worker
from lib.utils.utils_get_tiles import start_get_tiles


# "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
# respectively, to reorient during playback." - Author
# Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
# rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0


class GetTiles(Worker):
    def main(self):
        start_get_tiles()

    # def plot_test(self):
    #     # ctx.results[self.vid_proj][ctx.name][ctx.tiling][ctx.user]['frame'|'chunks']: list[list[str]] | d
    #     ctx.results = load_json(paths.get_tiles_json)
    #
    #     users_list = self.seen_tiles_metric[self.vid_proj][ctx.name]['1x1']['16'].keys()
    #
    #     for ctx.tiling in ctx.tiling_list:
    #         folder = self.seen_metrics_folder / f'1_{ctx.name}'
    #         folder.mkdir(parents=True, exist_ok=True)
    #
    #         for ctx.user in users_list:
    #             fig: plt.Figure
    #             fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
    #             ax = list(ax.flat)
    #             ax: list[plt.Axes]
    #
    #             for self.quality in self.quality_list:
    #                 result5 = defaultdict(list)    # By chunk
    #
    #                 for ctx.chunk in ctx.chunk_list:
    #                     seen_tiles_metric = self.seen_tiles_metric[self.vid_proj][ctx.name][ctx.tiling]
    #                     [self.quality][ctx.user][ctx.chunk]
    #                     tiles_list = seen_tiles_metric['time'].keys()
    #
    #                     result5[f'n_tiles'].append(len(tiles_list))
    #                     for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
    #                         value = [seen_tiles_metric[self.metric][tile] for tile in tiles_list]
    #                         percentile = list(np.percentile(value, [0, 25, 50, 75, 100]))
    #                         Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação
    #                         dos tiles)
    #                         result5[f'{self.metric}_sum'].append(np.sum(value))
    #                         tempo médio de um chunk (com decodificação paralela) (média dos tempos de decodificação
    #                         dos tiles)
    #                         result5[f'{self.metric}_avg'].append(np.average(value))
    #                         result5[f'{self.metric}_std'].append(np.std(value))
    #                         result5[f'{self.metric}_min'].append(percentile[0])
    #                         result5[f'{self.metric}_q1'].append(percentile[1])
    #                         result5[f'{self.metric}_median'].append(percentile[2])
    #                         result5[f'{self.metric}_q2'].append(percentile[3])
    #                         result5[f'{self.metric}_max'].append(percentile[4])
    #
    #                 ax[0].plot(result5['time_sum'], label=f'CRF{self.quality}')
    #                 ax[1].plot(result5['time_avg'], label=f'CRF{self.quality}')
    #                 ax[2].plot(result5['rate_sum'], label=f'CRF{self.quality}')
    #                 ax[3].plot(result5['PSNR_avg'], label=f'CRF{self.quality}')
    #                 ax[4].plot(result5['S-PSNR_avg'], label=f'CRF{self.quality}')
    #                 ax[5].plot(result5['WS-PSNR_avg'], label=f'CRF{self.quality}')
    #                 ax[6].plot(result5['n_tiles'], label=f'CRF{self.quality}')
    #
    #                 ax[0].set_title('time_sum')
    #                 ax[1].set_title('time_avg')
    #                 ax[2].set_title('rate_sum')
    #                 ax[3].set_title('PSNR_avg')
    #                 ax[4].set_title('S-PSNR_avg')
    #                 ax[5].set_title('WS-PSNR_avg')
    #                 ax[6].set_title('n_tiles')
    #
    #             for a in ax[:-1]:
    #                 a.legend(loc='upper right')
    #
    #             fig.suptitle(f'{self.video} {ctx.tiling} - user {ctx.user}')
    #             fig.suptitle(f'{self.video} {ctx.tiling} - user {ctx.user}')
    #             fig.tight_layout()
    #             # fig.show()
    #             img_name = folder / f'{ctx.tiling}_user{ctx.user}.png'
    #             fig.savefig(img_name)
    #             plt.close(fig)
    #

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
#         yaw_pitch_roll_iter = iter(ctx.hmd_dataset[ctx.name][ctx.user])
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
