from collections import Counter
from pathlib import Path
from time import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skvideo.io import FFmpegReader

<<<<<<< Updated upstream
from py360tools import ProjectionBase
=======
from .assets import Logger, Worker, AutoDict
from lib.assets.paths import Paths
from .py360tools import xyz2ea, ERP, CMP, compose
>>>>>>> Stashed changes
from lib.utils.util import load_json, save_json, lin_interpol, splitx, idx2xy

pi = np.pi
pi2 = np.pi * 2

# "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
# respectively, to reorient during playback." - Author
# Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
# rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0
rotation_map = {'cable_cam_nas': 265 / 180 * pi, 'drop_tower_nas': 180 / 180 * pi,
                'wingsuit_dubai_nas': 63 / 180 * pi, 'drone_chases_car_nas': 81 / 180 * pi}


class GetTilesPath(Worker, Logger, Paths):
    dataset_folder: Path
    video_id_map: dict
    user_map: dict
    _csv_dataset_file: Path
    video_name: str
    user_id: str
    head_movement: pd.DataFrame

    # <editor-fold desc="Dataset Path">
    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def dataset_folder(self) -> Path:
        return Path('datasets')

    @property
    def dataset_json(self) -> Path:
        return self.dataset_folder / f'{self.config["dataset_name"]}.json'

    @property
    def csv_dataset_file(self) -> Path:
        return self._csv_dataset_file

    @csv_dataset_file.setter
    def csv_dataset_file(self, value):
        self._csv_dataset_file = value
        user_nas_id, video_nas_id = self._csv_dataset_file.stem.split('_')
        self.video_name = self.video_id_map[video_nas_id]
        self.user_id = self.user_map[user_nas_id]

        names = ['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
        self.head_movement = pd.read_csv(self.csv_dataset_file, names=names)
    # </editor-fold>


class ProcessNasrabadi(GetTilesPath):
    dataset_final = AutoDict()
    previous_line: tuple
    frame_counter: int

    def main(self):
        print(f'Processing dataset {self.dataset_folder}.')
        if self.dataset_json.exists(): return

        self.video_id_map = load_json(f'{self.dataset_folder}/videos_map.json')
        self.user_map = load_json(f'{self.dataset_folder}/usermap.json')

        for self.csv_dataset_file in self.dataset_folder.glob('*/*.csv'):
            self.frame_counter = 0
            self.worker()

        print(f'Finish. Saving as {self.dataset_json}.')
        save_json(self.dataset_final, self.dataset_json)

    def worker(self):
        # For each  csv_file
        yaw_pitch_roll_frames = []
        start_time = time()
        n = 0

        print(f'\rUser {self.user_id} - {self.video_name} - ', end='')
        for n, line in enumerate(self.head_movement.itertuples(index=False, name=None)):
            timestamp, qx, qy, qz, qw, vx, vy, vz = map(float, line)
            xyz = np.array([vx, -vy, vz])  # Based on paper

            try:
                yaw_pitch_roll = self.process_vectors((timestamp, xyz))
                yaw_pitch_roll_frames.append(list(yaw_pitch_roll))
                self.frame_counter += 1
                if self.frame_counter == 1800: break
            except ValueError:
                pass
            self.previous_line = timestamp, xyz

        yaw_pitch_roll_frames += [yaw_pitch_roll_frames[-1]] * (1800 - len(yaw_pitch_roll_frames))

        self.dataset_final[self.video_name][self.user_id] = yaw_pitch_roll_frames
        print(f'Samples {n:04d} - {self.frame_counter=} - {time() - start_time:0.3f} s.')

    def process_vectors(self, actual_line):
        timestamp, xyz = actual_line
        frame_timestamp = self.frame_counter / 30

        if timestamp < frame_timestamp:
            # Skip. It's not the time.
            raise ValueError
        elif timestamp > frame_timestamp:
            # Linear Interpolation
            old_timestamp, old_xyz = self.previous_line
            xyz = lin_interpol(frame_timestamp, timestamp, old_timestamp, np.array(xyz), np.array(old_xyz))

        yaw, pitch = xyz2ea(xyz).T
        roll = [0] * len(yaw) if isinstance(yaw, np.ndarray) else 0

        if self.video_name in rotation_map:
            yaw -= rotation_map[self.video_name]

        yaw = np.mod(yaw + pi, pi2) - pi
        return np.round(np.array([yaw, pitch, roll]), 6).T


class GetTilesProps(GetTilesPath):
    get_tiles_data: dict
    results: AutoDict
    _dataset: dict
    _get_tiles_data: dict
    projection_dict: dict

    def make_projections(self, proj_res=('720x360', '540x360'), vp_shape=(294, 440)):
        self.projection_dict = AutoDict()
        for self.tiling in self.tiling_list:
            erp = ERP(tiling=self.tiling,
                      proj_res=proj_res[0],
                      vp_shape=vp_shape,
                      fov=self.fov)
            cmp = CMP(tiling=self.tiling,
                      proj_res=proj_res[1],
                      vp_shape=vp_shape,
                      fov=self.fov)
            self.projection_dict['erp'][self.tiling] = erp
            self.projection_dict['cmp'][self.tiling] = cmp

    @property
    def projection_obj(self) -> ProjectionBase:
        return self.projection_dict[self.proj][self.tiling]

    @property
    def dataset(self) -> dict:
        try:
            return self._dataset
        except AttributeError:
            self._dataset = load_json(self.dataset_json)
            return self._dataset

    @property
    def users_list(self) -> list[str]:
        return list(self.dataset[self.name].keys())

    @property
    def get_tiles_folder(self) -> Path:
        folder = self.project_path / 'get_tiles'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def get_tiles_json(self) -> Path:
        filename = f'get_tiles_{self.config["dataset_name"]}_{self.proj}_{self.name}_fov{self.fov}.json'
        path = self.get_tiles_folder / filename
        return path

    @property
    def counter_tiles_json(self):
        folder = self.get_tiles_folder / 'counter'
        folder.mkdir(parents=True, exist_ok=True)
        filename = f'counter_{self.config["dataset_name"]}_{self.proj}_{self.name}_fov{self.fov}.json'
        path = folder / filename
        return path

    @property
    def heatmap_tiling(self):
        folder = self.get_tiles_folder / 'heatmap'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'heatmap_tiling_{self.dataset_name}_{self.proj}_{self.name}_{self.tiling}_fov{self.fov}.png'
        return path


class GetTiles(GetTilesProps):
    _projection: ProjectionBase
    n_frames: int
    changed_flag: bool
    erp_list: dict[str, ProjectionBase]
    cmp_list: dict[str, ProjectionBase]
    tiles_1x1: dict[str, Union[dict[str, list[str]], list[list[str]]]]
    error: bool
    projection_dict: dict
    proj: str = None
    name: str = None
    tiling: str = None
    user: str = None

    def main(self):
        self.init()

        for self.proj in self.proj_list:
            for self.name in self.name_list:
                self.load_results()

                try:
                    for self.tiling in self.tiling_list:
                        self.load_projection()

                        for self.user in self.users_list:
                            start = time()
                            self.worker()
                            print(f'\ttime =  {time() - start}')
                finally:
                    self.save_results()

            # self.count_tiles()
            # self.heatmap()
            # self.plot_test()

    def load_projection(self):
        self._projection = self.projection_dict['erp'][self.tiling]

    def init(self):
        self.tiles_1x1 = {'frame': [["0"]] * self.n_frames,
                          'chunks': {str(i): ["0"] for i in range(1, int(self.duration) + 1)}}

        self.projection_dict = AutoDict()
        for self.tiling in self.tiling_list:
            if self.tiling == '1x1': continue
            erp = ERP(tiling=self.tiling, proj_res='1080x540', vp_shape=(540, 660), fov=self.fov)
            cmp = CMP(tiling=self.tiling, proj_res='810x540', vp_shape=(540, 660), fov=self.fov)
            self.projection_dict['erp'][self.tiling] = erp
            self.projection_dict['cmp'][self.tiling] = cmp

    def load_results(self):
        self.changed_flag = False

        try:
            self.results = load_json(self.get_tiles_json,
                                     object_hook=AutoDict)
        except FileNotFoundError:
            self.results = AutoDict()

    def save_results(self):
        if self.changed_flag:
            print('\n\tSaving.')
            save_json(self.results, self.get_tiles_json)

        pass

    def get_user_samples(self) -> AutoDict:
        """

        :return: {'frame': list,  # 1800 elements
                  'chunks': {str: list}} # 60 elements
        """
        return self.results[self.proj][self.name][self.tiling][self.user]

    def get_dataset_user_samples(self):
        return self.dataset[self.name][self.user]

    def worker(self):
        print(f'{self.proj} {self.name} {self.tiling} - User {self.user}')

        if self.get_user_samples() != {}:
            return
        elif not self.changed_flag:
            self.changed_flag = True

        if self.tiling == '1x1':
            self.get_user_samples().update(self.tiles_1x1)
            return

        result_frames = []
        for yaw_pitch_roll in self.get_dataset_user_samples():
            vptiles: list[str] = self._projection.get_vptiles(yaw_pitch_roll)
            result_frames.append(vptiles)

        result_chunks = {}
        tiles_in_chunk = set()
        for frame, vptiles in enumerate(result_frames):
            tiles_in_chunk.update(vptiles)

            if (frame + 1) % 30 == 0:
                chunk = frame // 30 + 1  # chunk start from 1
                result_chunks[f'{chunk}'] = list(tiles_in_chunk)
                tiles_in_chunk.clear()

        user_samples = self.get_user_samples()
        user_samples['frames'] = result_frames
        user_samples['chunks'] = result_chunks

    def count_tiles(self):
        if self.counter_tiles_json.exists(): return

        self.results = load_json(self.get_tiles_json)
        result = {}

        for self.tiling in self.tiling_list:
            if self.tiling == '1x1': continue

            # <editor-fold desc="Count tiles">
            tiles_counter_chunks = Counter()  # Collect tiling count

            for self.user in self.users_list:
                result_chunks: dict[str, list[str]] = self.results[self.proj][self.name][self.tiling][self.user][
                    'chunks']

                for chunk in result_chunks:
                    tiles_counter_chunks = tiles_counter_chunks + Counter(result_chunks[chunk])
            # </editor-fold>

            print(tiles_counter_chunks)
            dict_tiles_counter_chunks = dict(tiles_counter_chunks)

            # <editor-fold desc="Normalize Counter">
            nb_chunks = sum(dict_tiles_counter_chunks.values())
            for self.tile in self.tile_list:
                try:
                    dict_tiles_counter_chunks[self.tile] /= nb_chunks
                except KeyError:
                    dict_tiles_counter_chunks[self.tile] = 0
            # </editor-fold>

            result[self.tiling] = dict_tiles_counter_chunks

        save_json(result, self.counter_tiles_json)

    def heatmap(self):
        results = load_json(self.counter_tiles_json)

        for self.tiling in self.tiling_list:
            if self.tiling == '1x1': continue

            filename = f'heatmap_tiling_{self.dataset_name}_{self.proj}_{self.name}_{self.tiling}_fov{self.fov}.png'
            heatmap_tiling = (self.get_tiles_folder / filename)
            if heatmap_tiling.exists(): continue

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

    # def plot_test(self):
    #     # self.results[self.vid_proj][self.name][self.tiling][self.user]['frame'|'chunks']: list[list[str]] | d
    #     self.results = load_json(self.get_tiles_json)
    #
    #     users_list = self.seen_tiles_metric[self.vid_proj][self.name]['1x1']['16'].keys()
    #
    #     for self.tiling in self.tiling_list:
    #         folder = self.seen_metrics_folder / f'1_{self.name}'
    #         folder.mkdir(parents=True, exist_ok=True)
    #
    #         for self.user in users_list:
    #             fig: plt.Figure
    #             fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
    #             ax = list(ax.flat)
    #             ax: list[plt.Axes]
    #
    #             for self.quality in self.quality_list:
    #                 result5 = defaultdict(list)    # By chunk
    #
    #                 for self.chunk in self.chunk_list:
    #                     seen_tiles_metric = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling]
    #                     [self.quality][self.user][self.chunk]
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
    #             fig.suptitle(f'{self.video} {self.tiling} - user {self.user}')
    #             fig.suptitle(f'{self.video} {self.tiling} - user {self.user}')
    #             fig.tight_layout()
    #             # fig.show()
    #             img_name = folder / f'{self.tiling}_user{self.user}.png'
    #             fig.savefig(img_name)
    #             plt.close(fig)
    #


class TestGetTiles(GetTiles):
    def init(self):
        self.tiling_list.remove('1x1')
        self.quality = '28'
        self._get_tiles_data = {}
        self.make_projections()

    def main(self):
        self.init()

        for self.proj in [self.proj_list[1]]:
            for self.name in self.name_list:
                for self.tiling in self.tiling_list:
                    for self.user in [self.users_list[0]]:
                        self.worker()

    frame_n: int

    def worker(self, overwrite=False):
        print(f'{self.proj}, {self.name}, {self.tiling}, {self.user}')
        yaw_pitch_roll_iter = iter(self.dataset[self.name][self.user])
        self.frame_n = 0
        for self.chunk in self.chunk_list:
            for proj_frame in self.mount_chunk_frames():
                if self.output_video.exists():
                    print(f' Exist. Skipping.')
                    self.frame_n += 1
                    next(yaw_pitch_roll_iter)
                    continue

                self.projection_obj.yaw_pitch_roll = next(yaw_pitch_roll_iter)

                all_tiles_borders = self.projection_obj.draw_all_tiles_borders()
                vp_tiles = self.projection_obj.draw_vp_tiles()
                vp_mask = self.projection_obj.draw_vp_mask()
                try:
                    vp_borders = self.projection_obj.draw_vp_borders()
                except IndexError:
                    vp_borders = self.projection_obj.draw_vp_borders()

                viewport_array = self.projection_obj.get_vp_image(proj_frame)

                frame = compose(proj_frame_image=Image.fromarray(proj_frame),
                                all_tiles_borders_image=Image.fromarray(all_tiles_borders),
                                vp_tiles_image=Image.fromarray(vp_tiles),
                                vp_mask_image=Image.fromarray(vp_mask),
                                vp_borders_image=Image.fromarray(vp_borders),
                                vp_image=Image.fromarray(viewport_array),
                                )
                frame.save(self.output_video)
                self.frame_n += 1

        print('')

    def mount_chunk_frames(self, proj_shape=None, tile_shape=None,
                           tiling_shape=None, chunk=None, seen_tiles=None):
        proj_h, proj_w = self.projection_obj.proj_shape
        frame_proj = np.zeros((proj_h, proj_w, 3), dtype='uint8')
        seen_tiles = self.get_user_samples()['chunks'][self.chunk]
        tiles_reader = {self.tile: FFmpegReader(f'{self.segment_file}').nextFrame()
                        for self.tile in seen_tiles}
        # seen_tiles = projection.get_vptiles()  # by frame

        for frame in range(30):
            for self.tile in seen_tiles:
                tile_h, tile_w = self.projection_obj.tile_shape
                tile_m, tile_n = idx2xy(int(self.tile), (self.projection_obj.tiling_h, self.projection_obj.tiling_w))
                tile_x, tile_y = tile_m * tile_w, tile_n * tile_h

                tile_frame = next(tiles_reader[self.tile])

                tile_resized = Image.fromarray(tile_frame).resize((tile_w, tile_h))
                tile_resized_array = np.asarray(tile_resized)
                frame_proj[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w, :] = tile_resized_array
            yield frame_proj

    @property
    def output_video(self):
        folder = self.get_tiles_folder / 'videos' / f'{self.proj}_{self.name}_{self.tiling}'
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"user{self.user}_{self.frame_n}.png"

        return output


GetTilesOptions = {'0': ProcessNasrabadi,  # 0
                   '1': GetTiles,  # 1
                   '2': TestGetTiles,  # 1
                   }
