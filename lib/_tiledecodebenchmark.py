from collections import defaultdict
from logging import warning
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .assets import GlobalPaths, Config, Log, AutoDict, Bcolors, Utils, SiTi
from .transform import splitx
from .util import save_json, load_json, run_command, decode_file, get_times


class TileDecodeBenchmarkPaths(Utils, Log, GlobalPaths):
    # Folders
    original_folder = Path('original')
    lossless_folder = Path('lossless')
    compressed_folder = Path('compressed')
    segment_folder = Path('segment')
    _viewport_folder = Path('viewport')
    _siti_folder = Path('siti')
    _check_folder = Path('check')

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
    def dectime_result_json(self) -> Path:
        """
        By Video
        :return:
        """
        folder = self.project_path / self.dectime_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'time_{self.video}.json'

    @property
    def bitrate_result_json(self) -> Path:
        folder = self.project_path / self.segment_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'rate_{self.video}.json'

    @property
    def siti_folder(self):
        folder = self.project_path / self._siti_folder
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
        folder = self.project_path / self._siti_folder
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

    def __init__(self, config: str):
        self.config = Config(config)
        self.print_resume()
        with self.logger():
            self.main()

    def main(self):
        ...


class Prepare(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.videos_list:
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
        for self.video in self.videos_list:  # if self.video != 'chariot_race_erp_nas': continue
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
            print(f'{Bcolors.FAIL}The file {self.compressed_log} is corrupt. Skipping.{Bcolors.ENDC}')
            self.clean_compress()
            return False

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.log('CODEC ERROR', self.compressed_log)
            print(f'{Bcolors.FAIL}The file {self.compressed_log} have codec different of Lavc59.18.100 libx265. Skipping.{Bcolors.ENDC}')
            self.clean_compress()
            return False

        # decodifique os comprimidos
        if decode:
            stdout = decode_file(self.compressed_file)
            if "frame= 1800" not in stdout:
                print(f'{Bcolors.FAIL}Compress Decode Error. Cleaning.{Bcolors.ENDC}.')
                self.log(f'Compress Decode Error.', self.compressed_file)
                self.clean_compress()
                return False

        print(f'{Bcolors.FAIL}The file {self.compressed_file} is OK.{Bcolors.ENDC}')
        return True

    def clean_compress(self):
        self.compressed_log.unlink(missing_ok=True)
        self.compressed_file.unlink(missing_ok=True)


class Segment(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                with self.multi() as _:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            if self.skip(): return

                            self.worker()

    def worker(self) -> Any:
        print(f'==== Segment {self.compressed_file} ====')
        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
        # ime 1 -reset_timestamps 1 output%03d.mp4

        cmd = f'bash -k -c '
        cmd += '"bin/MP4Box '
        cmd += '-split 1 '
        cmd += f'{self.compressed_file.as_posix()} '
        cmd += f"-out {self.segments_folder.as_posix()}/tile{self.tile}_'$'num%03d$.mp4"
        cmd += f'|& tee {self.segment_log.as_posix()}"'

        self.command_pool.append(cmd)
        # run_command(cmd)

    def skip(self, decode=False):
        # first compressed file
        if not self.compressed_file.exists():
            self.log('compressed_file NOTFOUND.', self.compressed_file)
            print(f'{Bcolors.FAIL}The file {self.compressed_file} not exist. Skipping.{Bcolors.ENDC}')
            return True

        # second check segment log
        try:
            segment_log = self.segment_log.read_text()
        except FileNotFoundError:
            self.clean_segments()
            return False

        if 'file 60 done' not in segment_log:
            # self.compressed_file.unlink(missing_ok=True)
            # self.compressed_log.unlink(missing_ok=True)
            print(f'{Bcolors.FAIL}The file {self.segment_log} is corrupt. Processing.{Bcolors.ENDC}')
            self.log('Segment_log is corrupt. Cleaning', self.segment_log)
            self.clean_segments()
            return False

        # Se log está ok; verifique os segmentos.
        for self.chunk in self.chunk_list:
            # Se segmento existe
            try:
                segment_file_size = self.segment_file.stat().st_size
            except FileNotFoundError:
                # um segmento não existe e o Log diz que está ok. limpeza.
                print(f'{Bcolors.FAIL}Segmentlog is OK. The file not exist. Cleaning.{Bcolors.ENDC}')
                self.log(f'Segmentlog is OK. The file {self.segment_file} not exist.', self.segment_log)
                self.clean_segments()
                return False

            if segment_file_size == 0:
                # um segmento size 0 e o Log diz que está ok. limpeza.
                print(f'{Bcolors.FAIL}Segmentlog is OK. The file SIZE 0. Cleaning.{Bcolors.ENDC}')
                self.log(f'Segmentlog is OK. The file {self.segment_file} SIZE 0', self.segment_file)
                self.clean_segments()
                return False

            # decodifique os segmentos
            if decode:
                stdout = decode_file(self.segment_file)

                if "frame=   30" not in stdout:
                    print(f'{Bcolors.FAIL}Segment Decode Error. Cleaning.{Bcolors.ENDC}.')
                    self.log(f'Segment Decode Error.', self.segment_file)
                    self.clean_segments()
                    return False

        print(f'{Bcolors.FAIL}The {self.segment_log} IS OK. Skipping.{Bcolors.ENDC}')
        return True

    def clean_segments(self):
        self.segment_log.unlink(missing_ok=True)
        for self.chunk in self.chunk_list:
            self.segment_file.unlink(missing_ok=True)


class Decode(TileDecodeBenchmarkPaths):
    turn: int

    def main(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.turn in range(self.decoding_num):
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                self.worker()

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']

        try:
            quality_list.remove('0')
        except ValueError:
            pass

        return quality_list

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
                print(f'{Bcolors.WARNING} The segment not exist. '
                      f'{Bcolors.ENDC}')
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
    change_flag: bool

    def main(self):
        for self.video in self.videos_list:
            self.result_rate = AutoDict()
            self.change_flag = True
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.bitrate()

            if self.change_flag:
                save_json(self.result_rate, self.bitrate_result_json)

    def skip1(self, check_result=True):
        if self.bitrate_result_json.exists():
            self.change_flag = False
            print(f'\n[{self.vid_proj}][{self.video}] - The result_json exist.')
            if check_result:
                self.result_rate = load_json(self.bitrate_result_json,
                                             object_hook=AutoDict)
                return False
            return True
        return False

    def bitrate(self) -> Any:
        print(f'\r{self.state_str()} ', end='')

        try:
            chunk_size = self.segment_file.stat().st_size
        except FileNotFoundError:
            self.log('SEGMENT_FILE_NOT_FOUND', self.segment_file)
            return

        if chunk_size == 0:
            self.log('BITRATE==0', self.segment_file)
            return

        bitrate = 8 * chunk_size / self.chunk_dur
        value = self.result_rate[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]

        if value == bitrate:
            return
        elif not self.change_flag:
            self.change_flag = True

        self.result_rate[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = bitrate

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']

        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


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
    times: list

    def main(self):
        for self.video in self.videos_list:
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.get_dectime()

            if self.change_flag:
                save_json(self.result_times, self.dectime_result_json)

    def skip1(self, check_result=False):
        if self.dectime_result_json.exists():
            print(f'\n[{self.vid_proj}][{self.video}] - The result_json exist.')

            if check_result:
                self.change_flag = False
                self.result_times = load_json(self.dectime_result_json,
                                              object_hook=AutoDict)
                return False

            return True
        else:
            return False

    def skip2(self):
        if not self.dectime_log.exists():
            print(f'\n{Bcolors.FAIL}    The dectime log not exist. Skipping.'
                  f'{Bcolors.ENDC}')
            self.log('DECTIME_FILE_NOT_FOUND', self.dectime_log)
            return True
        return False

    def check_time(self):
        if len(self.times) < self.decoding_num:
            print(f'\n{Bcolors.WARNING}    The dectime is lower than {self.decoding_num}: {Bcolors.ENDC}')
            self.log(f'DECTIME_NOT_DECODED_ENOUGH_{len(self.times)}', self.dectime_log)

        if 0 in self.times:
            print(f'\n{Bcolors.WARNING}    0  found: {Bcolors.ENDC}')
            self.log('DECTIME_ZERO_FOUND', self.dectime_log)

    def get_dectime(self) -> Any:
        print(f'\r{self.state_str()} = ', end='')
        if self.skip2(): return

        content = self.dectime_log.read_text(encoding='utf-8')

        self.times = get_times(content)
        self.times = self.times[-self.decoding_num:]
        self.times = sorted(self.times)
        self.check_time()

        result_times = self.result_times[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]

        if self.change_flag is False and result_times != self.times:
            self.change_flag = True

        self.result_times[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = self.times
        print(f'{self.times}', end='')

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self.result_times = AutoDict()
        self.change_flag = True

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']

        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


class MakeSiti(TileDecodeBenchmarkPaths):
    def main(self):
        self.tiling = '1x1'
        self.quality = '28'
        self.tile = '0'

        # self.calc_siti()
        # self.calc_stats()
        # self.plot_siti()
        # self.scatter_plot_siti()

    def calc_siti(self):
        for self.video in self.videos_list:
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
            return

        for self.video in self.videos_list:
            siti_results = load_json(self.siti_results)
            si = siti_results[self.video]['si']
            ti = siti_results[self.video]['ti']
            bitrate = self.compressed_file.stat().st_size * 8 / 60

            siti_stats['group'].append(self.group)
            siti_stats['proj'].append(self.vid_proj)
            siti_stats['video'].append(self.video)
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
            for self.video in self.videos_list:
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
        fig_erp.savefig(self.project_path / self._siti_folder / 'scatter_ERP.png')

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
        fig_cmp.savefig(self.project_path / self._siti_folder / 'scatter_CMP.png')


class RenameAndCheck(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.videos_list:
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
