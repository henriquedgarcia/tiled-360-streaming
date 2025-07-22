from time import time

import numpy as np
from py360tools import xyz2ea

from lib.assets.autodict import AutoDict
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.worker import Worker
from lib.utils.util import save_json, load_json, lin_interpol

pi = np.pi
pi2 = np.pi * 2
rotation_map = {'cable_cam_nas': 265 / 180 * pi, 'drop_tower_nas': 180 / 180 * pi,
                'wingsuit_dubai_nas': 63 / 180 * pi, 'drone_chases_car_nas': 81 / 180 * pi}


class ProcessNasrabadi(Worker, MakeDecodablePaths):
    dataset_final = AutoDict()
    previous_line: tuple
    frame_counter: int

    def main(self):
        print(f'Processing dataset {self.dataset_folder}.')
        if self.dataset_folderet_json.exists(): return

        ctx.video_id_map = load_json(f'{segmenter_paths.dataset_folder}/videos_map.json')
        ctx.user_map = load_json(f'{segmenter_paths.dataset_folder}/usermap.json')

        for self.csv_dataset_file in segmenter_paths.dataset_folder.glob('*/*.csv'):
            self.frame_counter = 0
            self.worker()

        print(f'Finish. Saving as {segmenter_paths.dataset_json}.')
        save_json(self.dataset_final, segmenter_paths.dataset_json)

    def worker(self):
        # For each  csv_file
        yaw_pitch_roll_frames = []
        start_time = time()
        n = 0

        print(f'\rUser {ctx.user_id} - {ctx.video_name} - ', end='')
        for n, line in enumerate(ctx.head_movement.itertuples(index=False, name=None)):
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

        yaw, pitch = xyz2ea(xyz=xyz).T
        roll = [0] * len(yaw) if isinstance(yaw, np.ndarray) else 0

        if self.name in rotation_map:
            yaw -= rotation_map[self.name]

        yaw = np.mod(yaw + pi, pi2) - pi
        return np.round(np.array([yaw, pitch, roll]), 6).T
