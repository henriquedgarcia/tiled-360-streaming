from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from lib.assets.autodict import AutoDict
from lib.assets.ctxinterface import CtxInterface

from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.seen_tiles_paths import SeenTilesPaths
from lib.assets.paths.segmenterpaths import SegmenterPaths
from lib.assets.paths.userqualitypaths import UserQualityPaths
from .assets.paths.tilequalitypaths import ChunkQualityPaths
from .assets.errors import AbortError
# import lib.erp as v360
from .utils.util import print_error, save_json, load_json, get_nested_value

pi = np.pi
pi2 = np.pi * 2


class UserProjectionMetricsProps(CtxInterface):
    user_metrics: AutoDict
    time_data: dict
    rate_data: dict
    qlt_data: dict
    get_tiles_data: dict
    dectime_paths: DectimePaths
    segmenter_paths: SegmenterPaths
    get_tiles_paths: SeenTilesPaths
    tile_chunk_quality_paths: ChunkQualityPaths
    user_quality_paths: UserQualityPaths

    @property
    def seen_metrics_folder(self) -> Path:
        folder = self.config.project_folder / 'UserProjectionMetrics'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def seen_metrics_json(self) -> Path:
        filename = f'seen_metrics_{self.ctx.config.dataset_file.stem}_{self.name}_{self.projection}.json'
        return self.seen_metrics_folder / filename

    def get_get_tiles(self):
        try:
            tiles_list = self.get_tiles_data[self.name][self.projection][self.tiling][self.user]['chunks'][self.chunk]
        except (KeyError, AttributeError):
            self.get_tiles_data = load_json(self.get_tiles_paths.get_tiles_json)
            self.time_data = load_json(self.dectime_paths.dectime_result_json, object_hook=dict)
            self.rate_data = load_json(self.segmenter_paths.bitrate_result_json, object_hook=dict)
            self.qlt_data = load_json(self.tile_chunk_quality_paths.chunk_quality_json, object_hook=dict)
            tiles_list = self.get_tiles_data[self.name][self.projection][self.tiling][self.user]['chunks'][self.chunk]
        return tiles_list


class UserQuality(UserProjectionMetricsProps):

    def init(self):
        self.tile_chunk_quality_paths = ChunkQualityPaths(self.ctx)
        self.dectime_paths = DectimePaths(self.ctx)
        self.segmenter_paths = SegmenterPaths(self.ctx)
        self.get_tiles_paths = SeenTilesPaths(self.ctx)
        self.user_quality_paths = UserQualityPaths(self.ctx)

    def main(self):
        self.init()
        for _ in self.iterator():
            self.user_metrics = AutoDict()
            try:
                self.worker()
            except AbortError as e:
                print_error(f'\t{e.args[0]}')
                continue
            save_json(self.user_metrics, self.user_quality_paths.user_metrics_json)

    def iterator(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        for self.user in self.users_list:
                            yield

    def get_metrics(self):
        keys = [self.name, self.projection, self.quality, self.tiling, self.tile, self.chunk]
        dectime_val = get_nested_value(self.time_data, keys)
        bitrate_val = get_nested_value(self.rate_data, keys)
        quality_val = get_nested_value(self.qlt_data, keys)
        return dectime_val, bitrate_val, quality_val

    def worker(self):
        for self.chunk in self.chunk_list:
            chunk_metrics = defaultdict(list)

            for self.tile in self.get_get_tiles():
                dectime_val, bitrate_val, quality_val = self.get_metrics()
                chunk_metrics['time'].append(dectime_val)
                chunk_metrics['time_std'].append(float(np.std(dectime_val)))
                chunk_metrics['rate'].append(bitrate_val)
                chunk_metrics['ssim'].append(quality_val['ssim'])
                chunk_metrics['mse'].append(quality_val['mse'])
                chunk_metrics['smse'].append(quality_val['smse'])
                chunk_metrics['wsmse'].append(quality_val['wsmse'])

            self.user_metrics[f'{self.chunk}']['time_avg'] = float(np.average(chunk_metrics['time']))
            self.user_metrics[f'{self.chunk}']['time_max'] = float(np.max(chunk_metrics['time']))
            self.user_metrics[f'{self.chunk}']['time_sum'] = float(np.sum(chunk_metrics['time']))
            self.user_metrics[f'{self.chunk}']['time_std'] = float(np.std(chunk_metrics['time']))
            self.user_metrics[f'{self.chunk}']['time_std_avg'] = float(np.average(chunk_metrics['time_std']))
            self.user_metrics[f'{self.chunk}']['rate'] = float(np.average(chunk_metrics['rate']))
            self.user_metrics[f'{self.chunk}']['ssim'] = float(np.average(chunk_metrics['ssim']))
            self.user_metrics[f'{self.chunk}']['mse'] = float(np.average(chunk_metrics['mse']))
            self.user_metrics[f'{self.chunk}']['smse'] = float(np.average(chunk_metrics['smse']))
            self.user_metrics[f'{self.chunk}']['wsmse'] = float(np.average(chunk_metrics['wsmse']))

    def graphs1(self):
        # for each user plot quality in function of chunks
        def img_name():
            folder = self.seen_metrics_folder / f'1_{self.name}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f'{self.tiling}_user{self.user}.png'

        def loop_video_tiling_user():
            for self.name in self.name_list:
                for self.projection in self.projection_list:
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
                            self.user_metrics[self.projection][self.name][self.tiling][self.quality][self.user][
                                self.chunk]
                    except (KeyError, AttributeError):
                        self.user_metrics = load_json(self.user_metrics)
                        seen_tiles_metric = \
                            self.user_metrics[self.projection][self.name][self.tiling][self.quality][self.user][
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
            for self.name in self.name_list:
                self.user_metrics = load_json(self.user_metrics)
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
                            self.user_metrics[self.projection][self.name][self.tiling][self.quality][self.user][
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

            fig.suptitle(f'{self.name} - {self.projection} - {self.tiling}')
            fig.tight_layout()
            # fig.show()
            fig.savefig(img_name)
            img_name = img_name().parent / f'{self.tiling}_{self.name}_{self.projection}.png'
            fig.savefig(img_name)
            plt.close(fig)

            # result3[f'time_avg_total'].append(np.average(result4[f'time_total']))  # comparando entre usuários usamos o tempo médio
            # result3[f'time_avg_avg_sum'].append(np.sum(result4[f'time_avg_sum']))  # tempo médio sem paralelismo
            # result3[f'time_avg_avg'].append(np.average(result4[f'time_avg']))  # tempo total com decodificação paralela
            # result3[f'rate_total'].append(np.sum(result4[f'rate_sum']))  # taxa de bits sempre soma
            # result3[f'psnr_avg'].append(np.average(result4[f'PSNR_avg']))  # qualidade sempre é média
            # result3[f'ws_psnr_avg'].append(np.average(result4[f'WS-PSNR']))
            # result3[f's_psnr_avg'].append(np.average(result4[f'S-PSNR']))


def show(array):
    Image.fromarray(array).show()
