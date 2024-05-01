import os
from collections import defaultdict
from logging import warning
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .assets import GlobalPaths, Config, Log, AutoDict, Utils, print_error
from .siti import SiTi
from .util import save_json, load_json, run_command, decode_file, get_times, splitx

v = ['wingsuit_dubai_cmp_nas']
t = ['9x6', '12x8']


# v=['angel_falls_erp_nas', 'angel_falls_cmp_nas', 'blue_angels_cmp_nas', 'blue_angels_erp_nas', 'cable_cam_erp_nas', 'cable_cam_cmp_nas', 'chariot_race_cmp_nas', 'chariot_race_erp_nas', 'closet_tour_erp_nas', 'closet_tour_cmp_nas', 'drone_chases_car_erp_nas', 'drone_chases_car_cmp_nas', 'drone_footage_erp_nas', 'drone_footage_cmp_nas', 'drone_video_erp_nas', 'drone_video_cmp_nas', 'drop_tower_erp_nas', 'drop_tower_cmp_nas', 'dubstep_dance_erp_nas', 'dubstep_dance_cmp_nas', 'elevator_lift_erp_nas', 'elevator_lift_cmp_nas', 'glass_elevator_erp_nas', 'glass_elevator_cmp_nas', 'montana_erp_nas', 'montana_cmp_nas', 'motorsports_park_erp_nas', 'motorsports_park_cmp_nas', 'nyc_drive_erp_nas', 'nyc_drive_cmp_nas', 'pac_man_erp_nas', 'pac_man_cmp_nas', 'penthouse_erp_nas', 'penthouse_cmp_nas', 'petite_anse_erp_nas', 'petite_anse_cmp_nas', 'rhinos_erp_nas', 'rhinos_cmp_nas', 'sunset_erp_nas', 'sunset_cmp_nas', 'three_peaks_erp_nas', 'three_peaks_cmp_nas', 'video_04_erp_nas', 'video_04_cmp_nas', 'video_19_erp_nas', 'video_19_cmp_nas', 'video_20_erp_nas', 'video_20_cmp_nas', 'video_22_erp_nas', 'video_22_cmp_nas', 'video_23_erp_nas', 'video_23_cmp_nas', 'video_24_erp_nas', 'video_24_cmp_nas', 'wingsuit_dubai_erp_nas', 'wingsuit_dubai_cmp_nas']


class TileDecodeBenchmarkPaths(Utils, Log, GlobalPaths):
    # Folders

    @property
    def basename(self):
        return Path(f'{self.name}_'
                    f'{self.resolution}_'
                    f'{self.fps}_'
                    f'{self.tiling}_'
                    f'{self.rate_control}{self.quality}')

    @property
    def basename2(self):
        return Path(f'{self.name}_{self.resolution}_{self.fps}/'
                    f'{self.tiling}/'
                    f'{self.rate_control}{self.quality}/')

    @property
    def original_file(self) -> Path:
        return self.original_folder / self.original

    @property
    def lossless_file(self) -> Path:
        folder = self.project_path / self.lossless_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.name}_{self.resolution}_{self.fps}.mp4'

    @property
    def compressed_file(self) -> Path:
        folder = self.project_path / self.compressed_folder / self.basename2
        folder.absolute().mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.mp4'

    @property
    def compressed_log(self) -> Path:
        compressed_log = self.compressed_file.with_suffix('.log')
        return compressed_log

    @property
    def segments_folder(self) -> Path:
        folder = self.project_path / self.segment_folder / self.basename2 / f'tile{self.tile}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def segment_file(self) -> Path:
        chunk = int(str(self.chunk))
        return self.segments_folder / f'tile{self.tile}_{chunk:03d}.mp4'

    @property
    def segment_log(self) -> Path:
        return self.segments_folder / f'tile{self.tile}.log'

    @property
    def segment_reference_log(self) -> Path:
        qlt = self.quality
        self.quality = '0'
        segment_log = self.segment_log
        self.quality = qlt
        return segment_log

    @property
    def reference_segment(self) -> Union[Path, None]:
        qlt = self.quality
        self.quality = '0'
        segment_file = self.segment_file
        self.quality = qlt
        return segment_file

    @property
    def reference_compressed(self) -> Union[Path, None]:
        qlt = self.quality
        self.quality = '0'
        compressed_file = self.compressed_file
        self.quality = qlt
        return compressed_file

    @property
    def _dectime_folder(self) -> Path:
        folder = self.project_path / self.dectime_folder / self.basename2 / f'tile{self.tile}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def dectime_log(self) -> Path:
        chunk = int(str(self.chunk))
        return self._dectime_folder / f'chunk{chunk:03d}.log'

    @property
    def siti_folder(self):
        folder = self.project_path / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def siti_stats(self) -> Path:
        return self.siti_folder / f'siti_stats.csv'

    @property
    def siti_plot(self) -> Path:
        return self.siti_folder / f'siti_plot.png'

    @property
    def siti_results(self) -> Path:
        folder = self.project_path / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        name = f'siti_results'
        if self.video:
            name += f'_{self.video}'
        if self.tiling:
            name += f'_{self.tiling}'
        if self.quality:
            name += f'_{self.config["rate_control"]}{self.quality}'
        if self.tile:
            name += f'_tile{self.tile}'
        if self.chunk:
            name += f'_chunk{self.chunk}'

        return folder / f'siti_results_{self.video}_crf{self.quality}.csv'


    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']

        try:
            quality_list.remove('0')
        except ValueError:
            pass

        return quality_list


class Prepare(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.video_list:
            self.worker()

    def worker(self, overwrite=False):
        original_file: Path = self.original_file
        lossless_file: Path = self.lossless_file
        lossless_log: Path = self.lossless_file.with_suffix('.log')

        if lossless_file and not overwrite:
            warning(f'  The file {lossless_file=} exist. Skipping.')
            return

        if not original_file.exists():
            warning(f'  The file {original_file=} not exist. Skipping.')
            return

        resolution_ = splitx(self.resolution)
        dar = resolution_[0] / resolution_[1]

        cmd = f'bin/ffmpeg '
        cmd += f'-hide_banner -y '
        cmd += f'-ss {self.offset} '
        cmd += f'-i {original_file.as_posix()} '
        cmd += f'-crf 0 '
        cmd += f'-t {self.duration} '
        cmd += f'-r {self.fps} '
        cmd += f'-map 0:v '
        cmd += f'-vf scale={self.resolution},setdar={dar} '
        cmd += f'{lossless_file.as_posix()}'

        cmd = f'bash -c "{cmd}|& tee {lossless_log.as_posix()}"'
        print(cmd)
        run_command(cmd)


class Compress(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.video_list:  # if self.video != 'chariot_race_erp_nas': continue
            for self.tiling in self.tiling_list:
                with self.multi() as _:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            self.worker()

    def worker(self):
        if self.skip():
            return

        print(f'==== Processing {self.compressed_file} ====')
        x1, y1, x2, y2 = self.tile_position()

        factor = self.rate_control

        cmd = ['bin/ffmpeg -hide_banner -y -psnr']
        cmd += [f'-i {self.lossless_file.as_posix()}']
        cmd += [f'-c:v libx265']
        cmd += [f'-{factor} {self.quality} -tune psnr']
        cmd += [f'-x265-params']
        cmd += [f'keyint={self.gop}:'
                f'min-keyint={self.gop}:'
                f'open-gop=0:'
                f'scenecut=0:'
                f'info=0']
        if factor == 'qp':
            cmd[-1] += ':ipratio=1:pbratio=1'
        cmd += [f'-vf crop='
                f'w={x2 - x1}:h={y2 - y1}:'
                f'x={x1}:y={y1}']
        cmd += [f'{self.compressed_file.as_posix()}']
        cmd = ' '.join(cmd)

        cmd = f'bash -c "{cmd}&> {self.compressed_log.as_posix()}"'
        self.command_pool.append(cmd)

    def skip(self, decode=False):
        # first Lossless file
        if not self.lossless_file.exists():
            self.log(f'The lossless_file not exist.', self.lossless_file)
            print(f'The file {self.lossless_file} not exist. Skipping.')
            return True

        # second check compressed
        try:
            compressed_file_size = self.compressed_file.stat().st_size
        except FileNotFoundError:
            compressed_file_size = 0

        # third Check Logfile
        try:
            compressed_log_text = self.compressed_log.read_text()
        except FileNotFoundError:
            compressed_log_text = ''

        if compressed_file_size == 0 or compressed_log_text == '':
            self.clean_compress()
            return False

        if 'encoded 1800 frames' not in compressed_log_text:
            self.log('compressed_log is corrupt', self.compressed_log)
            print_error(f'The file {self.compressed_log} is corrupt. Skipping.')
            self.clean_compress()
            return False

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.log('CODEC ERROR', self.compressed_log)
            print_error(f'The file {self.compressed_log} have codec different of Lavc59.18.100 libx265. Skipping.')
            self.clean_compress()
            return False

        # decodifique os comprimidos
        if decode:
            stdout = decode_file(self.compressed_file)
            if "frame= 1800" not in stdout:
                print_error(f'Compress Decode Error. Cleaning..')
                self.log(f'Compress Decode Error.', self.compressed_file)
                self.clean_compress()
                return False

        print_error(f'The file {self.compressed_file} is OK.')
        return True

    def clean_compress(self):
        self.compressed_log.unlink(missing_ok=True)
        self.compressed_file.unlink(missing_ok=True)

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']
        if '0' not in quality_list:
            return ['0'] + quality_list
        return quality_list


class Segment(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.video_list:
            for self.tiling in self.tiling_list:
                with self.multi() as _:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            self.worker()

    def worker(self) -> Any:
        if self.skip(): return
        print(f'==== Segment {self.compressed_file} ====')
        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
        # ime 1 -reset_timestamps 1 output%03d.mp4
        cmd = 'bin/MP4Box '
        cmd += '-split 1 '
        cmd += f'{self.compressed_file.as_posix()} '
        cmd += f"-out {self.segments_folder.as_posix()}/tile{self.tile}_'$'num%03d$.mp4"
        cmd += f'2>&1 | tee {self.segment_log.as_posix()}'

        if os.name == 'nt':
            cmd = f'bash -c "{cmd}"'

        self.command_pool.append(cmd)
        # run_command(cmd)

    segment_log_txt: str

    def check_segment_log(self, decode=False):
        if 'file 60 done' not in self.segment_log_txt:
            # self.compressed_file.unlink(missing_ok=True)
            # self.compressed_log.unlink(missing_ok=True)
            print_error(f'The file {self.segment_log} is corrupt. Cleaning.')
            self.log('Segment_log is corrupt. Cleaning', self.segment_log)
            raise FileNotFoundError

        # Se log está ok; verifique os segmentos.
        for self.chunk in self.chunk_list:
            try:
                segment_file_size = self.segment_file.stat().st_size
            except FileNotFoundError as e:
                self.log(f'Segmentlog is OK, but segment not exist.', self.segment_log)
                print_error(f'Segmentlog is OK, but segment not exist. Cleaning.')
                raise e

            if segment_file_size == 0:
                # um segmento size 0 e o Log diz que está ok. limpeza.
                print_error(f'Segmentlog is OK. The file SIZE 0. Cleaning.')
                self.log(f'Segmentlog is OK. The file {self.segment_file} SIZE 0', self.segment_file)
                raise FileNotFoundError

            # decodifique os segmentos
            if decode:
                stdout = decode_file(self.segment_file)

                if "frame=   30" not in stdout:
                    print_error(f'Segment Decode Error. Cleaning..')
                    self.log(f'Segment Decode Error.', self.segment_file)
                    raise FileNotFoundError

    def read_segment_log(self):
        try:
            self.segment_log_txt = self.segment_log.read_text()
            self.check_segment_log()
        except FileNotFoundError as e:
            self.clean_segments()
            raise e

    def skip(self, decode=False):
        # first compressed file
        if not self.compressed_file.exists():
            self.log('compressed_file NOTFOUND.', self.compressed_file)
            print_error(f'The file {self.compressed_file} not exist. Skipping.')
            return True

        # second check segment log
        try:
            self.read_segment_log()
        except FileNotFoundError:
            return False

        print_error(f'The {self.segment_log} IS OK. Skipping.')
        return True

    def clean_segments(self):
        self.segment_log.unlink(missing_ok=True)
        for self.chunk in self.chunk_list:
            self.segment_file.unlink(missing_ok=True)

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']
        if '0' not in quality_list:
            return ['0'] + quality_list
        return quality_list


class Decode(TileDecodeBenchmarkPaths):
    turn: int

    def main(self):
        for self.video in self.video_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.turn in range(self.decoding_num):
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                self.worker()

    def clean_dectime_log(self):
        self.dectime_log.unlink(missing_ok=True)

    def skip(self):
        self.turn = 0
        try:
            content = self.dectime_log.read_text(encoding='utf-8')
            times = get_times(content)
            self.turn = len(times)
            if self.turn < self.decoding_num:
                raise FileNotFoundError
            print(f' Decoded {self.turn}.')
            return True
        except FileNotFoundError:
            if self.segment_file.exists():
                return False
            else:
                print_error(f'The segment not exist. ')
                self.log("segment_file not exist.", self.segment_file)
                return True

    def worker(self) -> Any:
        print(f'Decoding file "{self.segment_file}". ', end='')

        if self.skip():
            return

        print(f'Turn {self.turn + 1}')
        stdout = decode_file(self.segment_file, threads=1)
        with self.dectime_log.open('a') as f:
            f.write(f'\n==========\n{stdout}')
            print(' OK')


class GetBitrate(TileDecodeBenchmarkPaths):
    """
       The result dict have a following structure:
       results[video_name][tile_pattern][quality][tile_id][chunk_id]
               ['times'|'rate']
       [video_proj]    : The video projection
       [video_name]    : The video name
       [tile_pattern]  : The tile tiling. e.g. "6x4"
       [quality]       : Quality. An int like in crf or qp.
       [tile_id]           : the tile number. ex. max = 6*4
       [chunk_id]           : the chunk number. Start with 1.

    """
    turn: int
    _video: str
    result_rate: AutoDict
    error: bool
    change_flag: bool
    check_result = True

    def main(self):
        for self.video in self.video_list:
            if list(self.video_list).index(self.video) < 0: continue
            self.result_rate = AutoDict()
            self.change_flag = True
            self.error = False
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.bitrate()

            if self.change_flag and not self.error:
                print('Saving.')
                save_json(self.result_rate, self.bitrate_result_json)

    def skip1(self, ):
        if self.bitrate_result_json.exists():
            print(f'\n[{self.proj}][{self.video}] - The bitrate_result_json exist.')
            if self.check_result:
                self.change_flag = False
                self.result_rate = load_json(self.bitrate_result_json,
                                             object_hook=AutoDict)
                return False
            return True
        return False

    def bitrate(self) -> Any:
        print(f'\r{self.state_str()}: ', end='')

        try:
            bitrate = self.get_bitrate()
        except FileNotFoundError:
            self.error = True
            self.log('BITRATE FILE NOT FOUND', self.dectime_log)
            print_error(f'\n\n\tThe segment not exist. Skipping.')
            return

        if self.check_result:
            old_value = self.result_rate[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            if old_value == bitrate:
                return
            elif not self.change_flag:
                self.change_flag = True

        self.result_rate[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = bitrate
        print(f'{self.bitrate}', end='')

    def get_bitrate(self):
        chunk_size = self.segment_file.stat().st_size

        if chunk_size == 0:
            self.log('BITRATE==0', self.segment_file)
            self.segment_file.unlink()
            raise FileNotFoundError

        bitrate = 8 * chunk_size / self.chunk_dur

        return bitrate


class GetDectime(TileDecodeBenchmarkPaths):
    """
       The result dict have a following structure:
       results[video_name][tile_pattern][quality][tile_id][chunk_id]
               ['times'|'rate']
       [video_proj]    : The video projection
       [video_name]    : The video name
       [tile_pattern]  : The tile tiling. e.g. "6x4"
       [quality]       : Quality. An int like in crf or qp.
       [tile_id]           : the tile number. ex. max = 6*4
       [chunk_id]           : the chunk number. Start with 1.

       'times': list(float, float, float)
       'rate': float
    """
    result_times: AutoDict
    change_flag: bool
    check_result = True
    error: bool

    def main(self):
        for self.video in self.video_list:
            if list(self.video_list).index(self.video) < 38: continue
            self.result_times = AutoDict()
            self.change_flag = True
            self.error = False
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.dectime()

            if self.change_flag and not self.error:
                print('Saving.')
                save_json(self.result_times, self.dectime_result_json)

    def dectime(self) -> Any:
        print(f'\r{self.state_str()} = ', end='')

        try:
            times = self.get_dectime()
        except FileNotFoundError:
            self.error = True
            self.log('DECTIME_FILE_NOT_FOUND', self.dectime_log)
            print_error(f'\n\tThe dectime log not exist. Skipping.')
            return

        if self.check_result:
            old_value = self.result_times[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            if old_value == times:
                return
            elif not self.change_flag:
                self.change_flag = True

        self.result_times[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = times
        print(f'{times}', end='')

    def skip1(self):
        if self.dectime_result_json.exists():
            print(f'\n[{self.proj}][{self.video}] - The dectime_result_json exist.')
            if self.check_result:
                self.change_flag = False
                self.result_times = load_json(self.dectime_result_json,
                                              object_hook=AutoDict)
                return False
            return True
        return False

    def get_dectime(self) -> Any:
        content = self.dectime_log.read_text(encoding='utf-8')
        times = get_times(content)
        times = times[-self.decoding_num:]
        times = sorted(times)

        if len(times) < self.decoding_num:
            print_error(f'\n    The dectime is lower than {self.decoding_num}: ')
            self.log(f'DECTIME_NOT_DECODED_ENOUGH_{len(times)}', self.dectime_log)

        if 0 in times:
            print_error(f'\n    0  found: ')
            self.log('DECTIME_ZERO_FOUND', self.dectime_log)

        return times

    def skip2(self):
        if not self.dectime_log.exists():
            return True
        return False


class MakeSiti(TileDecodeBenchmarkPaths):
    def main(self):
        self.tiling = '1x1'
        self.quality = '28'
        self.tile = '0'

        # self.calc_siti()
        self.calc_stats()
        # self.plot_siti()
        # self.scatter_plot_siti()

    def calc_siti(self):
        for self.video in self.video_list:
            if self.siti_results.exists():
                print(f'siti_results FOUND {self.siti_results}. Skipping.')
                continue

            if not self.compressed_file.exists():
                self.log('compressed_file NOT_FOUND', self.compressed_file)
                print(f'compressed_file not exist {self.compressed_file}. Skipping.')
                continue

            siti = SiTi(self.compressed_file)

            siti_results_df = pd.DataFrame(siti.siti)
            siti_results_df.to_csv(self.siti_results)

    def calc_stats(self):
        siti_stats = defaultdict(list)
        if self.siti_stats.exists():
            print(f'{self.siti_stats} - the file exist')
            def calc():
                siti_stats = pd.read_csv(self.siti_stats)
                siti_stats1 = siti_stats[['group', 'name', 'proj', 'si_med', 'ti_med', 'bitrate']]
                # siti_stats2 = siti_stats1.sort_values('name').sort_values('proj').sort_values('group')
                midx = pd.MultiIndex.from_frame(siti_stats1[['group', 'name', 'proj']])
                data = siti_stats1[['si_med', 'ti_med', 'bitrate']]
                siti_stats3 = pd.DataFrame(data.values, index=midx)



            # return

        for self.video in self.video_list:
            siti_results = pd.read_csv(self.siti_results, index_col=0)
            si = siti_results['si']
            ti = siti_results['ti']
            bitrate = self.compressed_file.stat().st_size * 8 / 60

            siti_stats['group'].append(self.group)
            siti_stats['proj'].append(self.proj)
            siti_stats['video'].append(self.video)
            siti_stats['name'].append(self.name)
            siti_stats['tiling'].append(self.tiling)
            siti_stats['quality'].append(self.quality)
            siti_stats['tile'].append(self.tile)
            siti_stats['bitrate'].append(bitrate)

            siti_stats['si_avg'].append(np.average(si))
            siti_stats['si_std'].append(np.std(si))
            siti_stats['si_max'].append(np.max(si))
            siti_stats['si_min'].append(np.min(si))
            siti_stats['si_med'].append(np.median(si))
            siti_stats['ti_avg'].append(np.average(ti))
            siti_stats['ti_std'].append(np.std(ti))
            siti_stats['ti_max'].append(np.max(ti))
            siti_stats['ti_min'].append(np.min(ti))
            siti_stats['ti_med'].append(np.median(ti))

        pd.DataFrame(siti_stats).to_csv(self.siti_stats, index=False)

    def plot_siti(self):

        def plot1():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
            for self.video in self.video_list:
                siti_results = load_json(self.siti_results)
                name = self.name.replace('_nas', '')
                si = siti_results[self.video]['si']
                ti = siti_results[self.video]['ti']
                ax1.plot(si, label=name)
                ax2.plot(ti, label=name)

            ax1.set_xlabel("Frame")
            ax1.set_ylabel("Spatial Information")

            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Temporal Information')

            handles, labels = ax1.get_legend_handles_labels()
            fig.suptitle('SI/TI by frame')
            fig.legend(handles, labels,
                       loc='upper left',
                       bbox_to_anchor=[0.8, 0.93],
                       fontsize='small')
            fig.tight_layout()
            fig.subplots_adjust(right=0.78)
            fig.savefig(self.siti_plot)
            fig.show()

        def plot2():
            proj_list = ['erp', 'cmp']
            for name in self.name_list:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
                for proj in proj_list:
                    self.video = name.replace('_nas', f'_{proj}_nas')

                    siti_results_df = pd.read_csv(self.siti_results)
                    si = siti_results_df['si']
                    ti = siti_results_df['ti']
                    ax1.plot(si, label=self.video)
                    ax2.plot(ti, label=self.video)

                ax1.set_xlabel("Frame")
                ax1.set_ylabel("Spatial Information")

                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Temporal Information')

                fig.suptitle(f'SI/TI by frame - {name}')
                fig.legend(loc='upper left',
                           bbox_to_anchor=[0.8, 0.93],
                           fontsize='small')
                fig.tight_layout()
                fig.subplots_adjust(right=0.78)
                fig.savefig(self.siti_folder / f'siti_plot_{name}.png')
                fig.show()

        plot1()
        plot2()

    def scatter_plot_siti(self):
        siti_stats = pd.read_csv(self.siti_stats)
        change_name = lambda x: x.replace('_nas', '')
        siti_stats['video'].apply(change_name)

        si_max = siti_stats['si_med'].max()
        ti_max = siti_stats['ti_med'].max()

        siti_erp = siti_stats['proj'] == 'erp'
        siti_stats_erp = siti_stats[siti_erp][['video', 'si_med', 'ti_med']]
        fig_erp, ax_erp = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

        for idx, (video, si, ti) in siti_stats_erp.iterrows():
            ax_erp.scatter(si, ti, label=video + ' ')

        ax_erp.set_xlabel("Spatial Information")
        ax_erp.set_ylabel("Temporal Information")
        ax_erp.set_xlim(xmax=si_max + 5, xmin=0)
        ax_erp.set_ylim(ymax=ti_max + 5, ymin=0)
        ax_erp.legend(loc='upper left',
                      bbox_to_anchor=(1.01, 1.0),
                      fontsize='small')

        fig_erp.suptitle('ERP - SI x TI')
        fig_erp.tight_layout()
        fig_erp.show()
        fig_erp.savefig(self.project_path / self.siti_folder / 'scatter_ERP.png')

        ############################################
        siti_cmp = siti_stats['proj'] == 'cmp'
        siti_stats_cmp = siti_stats[siti_cmp][['video', 'si_med', 'ti_med']]
        fig_cmp, ax_cmp = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

        for idx, (video, si, ti) in siti_stats_cmp.iterrows():
            ax_cmp.scatter(si, ti, label=video)

        ax_cmp.set_xlabel("Spatial Information")
        ax_cmp.set_ylabel("Temporal Information")
        ax_cmp.set_xlim(xmax=si_max + 5, xmin=0)
        ax_cmp.set_ylim(ymax=ti_max + 5, ymin=0)
        ax_cmp.legend(loc='upper left',
                      bbox_to_anchor=(1.01, 1.0),
                      fontsize='small')

        fig_cmp.suptitle('CMP - SI x TI')
        fig_cmp.tight_layout()
        fig_cmp.show()
        fig_cmp.savefig(self.project_path / self.siti_folder / 'scatter_CMP.png')


class RenameAndCheck(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.video_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.turn in range(self.decoding_num):
                        for self.tile in self.tile_list:
                            self.compressed()

                            for self.chunk in self.chunk_list:
                                self.segmented()
                                pass

    def compressed(self) -> Any:
        folder = self.project_path / self.compressed_folder / self.basename
        old_name = folder / f'tile{self.tile}.mp4'
        if folder.exists():
            if old_name.exists() and not self.compressed_file.exists():
                old_name.replace(self.compressed_file)
                old_name.with_suffix('.log').replace(self.compressed_log)

            try:
                folder.rmdir()
            except OSError:
                pass

    def segmented(self):
        folder = self.project_path / self.segment_folder / self.basename
        if folder.exists():
            old_name = folder / f'tile{self.tile}.log'

            if old_name.exists() and not self.segment_log.exists():
                old_name.replace(self.segment_log)

            for self.chunk in self.chunk_list:
                chunk = int(str(self.chunk))
                old_file = folder / f'tile{self.tile}_{chunk:03d}.mp4'
                if old_file.exists() and not self.segment_file.exists():
                    old_file.replace(self.segment_file)

            try:
                folder.rmdir()
            except OSError:
                pass


TileDecodeBenchmarkOptions = {'0': Prepare,  # 0
                              '1': Compress,  # 1
                              '2': Segment,  # 2
                              '3': Decode,  # 3
                              '4': GetBitrate,  # 4
                              '5': GetDectime,  # 5
                              '6': MakeSiti,  # 6
                              }
