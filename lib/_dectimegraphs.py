from contextlib import contextmanager
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from cycler import cycler
from fitter import Fitter

from .assets import AutoDict, Bcolors, GlobalPaths, Utils
from .util import load_json, save_json, save_pickle, load_pickle


class DectimeGraphsPaths(GlobalPaths):
    error_type: str
    n_dist = 6
    bins = 30
    stats = defaultdict(list)
    correlations_bucket = defaultdict(list)
    workfolder_name: str
    dists_colors = {'burr12': 'tab:blue',
                    'fatiguelife': 'tab:orange',
                    'gamma': 'tab:green',
                    'invgauss': 'tab:red',
                    'rayleigh': 'tab:purple',
                    'lognorm': 'tab:brown',
                    'genpareto': 'tab:pink',
                    'pareto': 'tab:gray',
                    'halfnorm': 'tab:olive',
                    'expon': 'tab:cyan'}

    @property
    def workfolder(self) -> Path:
        folder = self.project_path / self.graphs_folder / self.workfolder_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_data(self) -> Path:
        folder = self.workfolder / 'data'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_fitter(self) -> Path:
        folder = self.workfolder / 'fitter'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_pdf_cdf(self) -> Path:
        folder = self.workfolder / 'pdf_cdf'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_boxplot(self) -> Path:
        folder = self.workfolder / 'boxplot'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def seen_tiles_data_file(self) -> Path:
        """
        Need None
        :return:
        """
        path = self.workfolder / f'seen_tiles_fov{self.fov}.json'
        return path

    @property
    def stats_file(self) -> Path:
        """
        Need bins
        :return:
        """
        stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
        return stats_file

    @property
    def correlations_file(self) -> Path:
        """
        Need None
        :return:
        """
        correlations_file = self.workfolder / f'correlations.json'
        return correlations_file


class DectimeGraphsObj(DectimeGraphsPaths, Utils):
    video_data_metric = {}
    metric_label = {'time': {'scilimits': (-3, -3),
                             'xlabel': f'Average Decoding Time (ms)'},
                    'time_std': {'scilimits': (-3, -3),
                                 'xlabel': f'Std Dev Decoding Time (ms)'},
                    'rate': {'scilimits': (6, 6),
                             'xlabel': f'Average Bit Rate (ms)'},
                    'MSE': {'scilimits': (0, 0),
                            'xlabel': f'Average PSNR'},
                    'SSIM': {'scilimits': (0, 0),
                             'xlabel': f'Average SSIM'},
                    'WS-MSE': {'scilimits': (0, 0),
                               'xlabel': f'Average WS-PSNR'},
                    'S-MSE': {'scilimits': (0, 0),
                              'xlabel': f'Average S-PSNR'}
                    }

    def __init__(self, config: str):
        self.rc_config()
        self.error_type = 'sse'

        super().__init__(config)

    @staticmethod
    def find_dist(dist_name, params):
        if dist_name == 'burr12':
            return dict(name='Burr Type XII',
                        parameters=f'c={params[0]}, d={params[1]}',
                        loc=params[2],
                        scale=params[3])
        elif dist_name == 'fatiguelife':
            return dict(name='Birnbaum-Saunders',
                        parameters=f'c={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'gamma':
            return dict(name='Gamma',
                        parameters=f'a={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'invgauss':
            return dict(name='Inverse Gaussian',
                        parameters=f'mu={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'rayleigh':
            return dict(name='Rayleigh',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        elif dist_name == 'lognorm':
            return dict(name='Log Normal',
                        parameters=f's={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'genpareto':
            return dict(name='Generalized Pareto',
                        parameters=f'c={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'pareto':
            return dict(name='Pareto Distribution',
                        parameters=f'b={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'halfnorm':
            return dict(name='Half-Normal',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        elif dist_name == 'expon':
            return dict(name='Exponential',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        else:
            raise ValueError(f'Distribution unknown: {dist_name}')

    @staticmethod
    def rc_config():
        rc_param = {"figure": {'figsize': (7.0, 1.2), 'dpi': 600, 'autolayout': True},
                    "axes": {'linewidth': 0.5, 'titlesize': 8, 'labelsize': 6,
                             'prop_cycle': cycler(color=[plt.get_cmap('tab20')(i) for i in range(20)])},
                    "xtick": {'labelsize': 6},
                    "ytick": {'labelsize': 6},
                    "legend": {'fontsize': 6},
                    "font": {'size': 6},
                    "patch": {'linewidth': 0.5, 'edgecolor': 'black', 'facecolor': '#3297c9'},
                    "lines": {'linewidth': 0.5, 'markersize': 2},
                    "errorbar": {'capsize': 4},
                    "boxplot": {'flierprops.marker': '+', 'flierprops.markersize': 1, 'flierprops.linewidth': 0.5,
                                'boxprops.linewidth': 0.0,
                                'capprops.linewidth': 1,
                                'medianprops.linewidth': 0.5,
                                'whiskerprops.linewidth': 0.5,
                                }
                    }

        for group in rc_param:
            mpl.rc(group, **rc_param[group])

    def process_value(self, value: Any, metric=None) -> float:
        if metric is not None:
            self.metric = metric
        # Process value according the metric
        if self.metric == 'time':  # value: list
            new_value = float(np.round(np.average(value), decimals=3))
        elif self.metric == 'time_std':  # value: list
            new_value = float(np.round(np.std(value), decimals=6))
        elif self.metric == 'rate':  # value: float
            new_value = float(value)
        else:  # if self.metric in ['MSE', 'WS-MSE', 'S-MSE']:
            metric_value = value[self.metric]  # value == dict[list]
            new_value = float(np.round(np.average(metric_value), decimals=3))
        return new_value

    def context_metric_proj_tiling(self):
        for self.metric in self.metric_list:
            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    yield

    def context_metric_proj_tiling_quality(self):
        for _ in self.context_metric_proj_tiling():
            for self.quality in self.quality_list:
                yield


class ByPatternProps(DectimeGraphsObj):
    results: dict
    data_bucket: dict
    error_type: str
    _fig_pdf: dict
    _fig_cdf: dict
    _fig_boxplot: dict
    fitter: Fitter
    workfolder_name = 'ByPattern'

    @property
    def chunk_results(self):
        results = self.results
        for state in self.state:
            results = results[state]
        return results

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


    @property
    def fitter_pickle_file(self) -> Path:
        """
        Need: metric, proj, tiling, and bins

        :return:  Path(fitter_pickle_file)
        """
        fitter_file = self.workfolder_fitter / f'fitter_{self.metric}_{self.proj}_{self.tiling}_{self.bins}bins.pickle'
        return fitter_file

    @property
    def data_bucket_file(self) -> Path:
        path = self.workfolder_data / f'data_bucket.json'
        return path

    @property
    def boxplot_file(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        folder = self.workfolder_boxplot
        mid = self.metric_list.index(self.metric)
        img_file = folder / f'boxplot_pattern_{mid}_{self.metric}_{self.proj}.png'
        return img_file

    @property
    def pdf_file(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        return self.workfolder_pdf_cdf / f'pdf_{self.metric}.png'

    @property
    def cdf_file(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        return self.workfolder_pdf_cdf / f'cdf_{self.metric}.png'

    @property
    def fig_pdf(self) -> figure.Figure:
        key = (self.metric,)
        try:
            fig = self._fig_pdf[key]
        except (KeyError, AttributeError):
            self._fig_pdf = {}
            fig = plt.figure(figsize=(12.0, 4),
                             dpi=600,
                             linewidth=0.5,
                             tight_layout=True)
            self._fig_pdf[key] = fig
        return fig

    @property
    def fig_cdf(self) -> figure.Figure:
        key = (self.metric,)
        try:
            fig = self._fig_cdf[key]
        except (KeyError, AttributeError):
            self._fig_cdf = {}
            fig = self._fig_cdf[key] = plt.figure(figsize=(12.0, 4),
                                                  dpi=600,
                                                  linewidth=0.5,
                                                  tight_layout=True)
        return fig

    @property
    def fig_boxplot(self) -> figure.Figure:
        # make an image for each metric and projection
        key = (self.metric,)
        try:
            _fig_boxplot = self._fig_boxplot[key]
        except (KeyError, AttributeError):
            self._fig_boxplot = {}
            _fig_boxplot = self._fig_boxplot[key] = plt.figure(figsize=(6.0, 4),
                                                               dpi=600,
                                                               linewidth=0.5,
                                                               tight_layout=True)
        return _fig_boxplot

    @property
    def json_metrics(self):
        return {'rate': self.bitrate_result_json,
                'time': self.dectime_result_json,
                'time_std': self.dectime_result_json,
                'SSIM': self.quality_result_json,
                'MSE': self.quality_result_json,
                'WS-MSE': self.quality_result_json,
                'S-MSE': self.quality_result_json
                }[self.metric]


class ByPatternMakeBucket(ByPatternProps):
    """
    [metric][proj][tiling] = [video, quality, tile, chunk]
    1x1 - 10.080 chunks - 1 tiles por tiling
    3x2 - 60.480 chunks - 6 tiles por tiling
    6x4 - 241.920 chunks - 24 tiles por tiling
    9x6 - 544.320 chunks - 54 tiles por tiling
    12x8 - 967.680 chunks - 96 tiles por tiling
    total - 1.824.480 chunks - 181 tiles
    """
    tiles_num = {"1x1": 10080,
                 "3x2": 60480,
                 "6x4": 241920,
                 "9x6": 544320,
                 "12x8": 967680}

    def main(self):
        print(f'====== ByPattern - error_type={self.error_type}, n_dist={self.n_dist} ======')
        if self.data_bucket_file.exists(): return
        self.data_bucket = AutoDict()
        for _ in self.main_context():
            chunk_data1 = self.video_data_metric[self.metric][self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            chunk_data2 = self.process_value(chunk_data1)

            try:
                self.data_bucket[self.metric][self.proj][self.tiling].append(chunk_data2)
            except AttributeError:
                self.data_bucket[self.metric][self.proj][self.tiling] = [chunk_data2]

        self.test()
        self.save()

    def load_all_metric(self):
        self.video_data_metric['rate'] = load_json(self.bitrate_result_json)
        self.video_data_metric['time'] = load_json(self.dectime_result_json)
        self.video_data_metric['time_std'] = self.video_data_metric['time']
        self.video_data_metric['SSIM'] = load_json(self.quality_result_json)
        self.video_data_metric['MSE'] = self.video_data_metric['SSIM']
        self.video_data_metric['WS-MSE'] = self.video_data_metric['SSIM']
        self.video_data_metric['S-MSE'] = self.video_data_metric['SSIM']

    def main_context(self):
        for self.video in self.video_list:
            self.load_all_metric()
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    print(f'\r\t{self.video} {self.tiling} {self.quality_str}      ', end='')
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            for self.metric in self.metric_list:
                                yield

    def test(self):
        for _ in self.context_metric_proj_tiling():
            bucket = self.data_bucket[self.metric][self.proj][self.tiling]
            if self.tiles_num[self.tiling] != len(bucket):
                self.log(f'bucket size error', self.data_bucket_file)
                raise ValueError(f'bucket size error')

    def save(self):
        print(f'\tSaving ... ', end='')
        save_json(self.data_bucket, self.data_bucket_file)
        print(f'Finished.')


class ByPatternFit(ByPatternProps):
    def main(self):
        print(f'====== ByPattern - error_type={self.error_type}, n_dist={self.n_dist} ======')
        self.data_bucket = load_json(self.data_bucket_file)
        for _ in self.context_metric_proj_tiling():
            if self.fitter_pickle_file.exists(): continue
            print(f'\tFitting - {self.metric} - {self.proj} - {self.tiling}...')
            samples = self.data_bucket[self.metric][self.proj][self.tiling]
            self.fitter = Fitter(samples, bins=self.bins, distributions=self.config['distributions'], timeout=1500)
            self.fitter.fit()
            print(f'\tSaving Fitter... ')
            save_pickle(self.fitter, self.fitter_pickle_file)

class ByPattern(ByPatternProps):
    def main(self):
        print(f'====== ByPattern - error_type={self.error_type}, n_dist={self.n_dist} ======')
        # Script para calcular a maior variação de tempo de decodificação entre as 5 decodificações.
        # time_var_rate = set()
        # for self.metric in ['time']:
        #     for self.proj in self.proj_list:
        #         for self.tiling in self.tiling_list:
        #             for self.name in self.name_list:
        #                 print(f'{self.metric}-{self.proj}-{self.tiling}-{self.name}')
        #                 vid_data = load_json(self.json_metrics)
        #                 for self.quality in self.quality_list:
        #                     for self.tile in self.tile_list:
        #                         for self.chunk in self.chunk_list:
        #                             chunk_data1 = vid_data[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
        #
        #                             time_var_rate.update([100*(1 - np.min(chunk_data1) / np.max(chunk_data1))])
        # max_time_var = max(list(time_var_rate))
        # for self.metric in self.metric_list:
        #     self.get_data_bucket()
        #     self.make_fit()
        #     self.make_hist()  # compare tiling
        #     self.make_boxplot()
        # self.calc_stats()
        # self.calc_corr()
        self.calc_corr2()

    def get_data_bucket(self):
        # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
        # 1x1 - 10.080 chunks - 1/181 tiles por tiling
        # 3x2 - 60.480 chunks - 6/181 tiles por tiling
        # 6x4 - 241.920 chunks - 24/181 tiles por tiling
        # 9x6 - 544.320 chunks - 54/181 tiles por tiling
        # 12x8 - 967.680 chunks - 96/181 tiles por tiling
        # total - 1.824.480 chunks - 181/181 tiles por tiling
        tiles_num = {"1x1": 10080,
                     "3x2": 60480,
                     "6x4": 241920,
                     "9x6": 544320,
                     "12x8": 967680}

        try:
            self.data_bucket = load_json(self.data_bucket_file)
            return
        except FileNotFoundError:
            self.data_bucket = AutoDict()

        for self.video in self.video_list:
            video_data_metric = load_json(self.json_metrics)
            for self.tiling in self.tiling_list:
                print(f'\r\t{self.metric} {self.video} {self.tiling}. len = ', end='')

                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            chunk_data1 = video_data_metric[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                            chunk_data2 = self.process_value(chunk_data1)
                            try:
                                self.data_bucket[self.metric][self.proj][self.tiling].append(chunk_data2)
                            except AttributeError:
                                self.data_bucket[self.metric][self.proj][self.tiling] = [chunk_data2]

                bucket = self.data_bucket[self.metric][self.proj][self.tiling]
                print(len(bucket))

        # test
        for proj in self.proj_list:
            for tiling in self.tiling_list:
                bucket = self.data_bucket[self.metric][proj][tiling]
                if tiles_num[tiling] != len(bucket):
                    self.log(f'bucket size error', self.data_bucket_file)
                    raise ValueError(f'bucket size error')



    def make_fit(self):
        # "deve salvar o arquivo"
        #                                56  x    6   x  var  x 60
        # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
        # 1x1 - 10014 chunks - 1/181 tiles por tiling
        # 3x2 - 60084 chunks - 6/181 tiles por tiling
        # 6x4 - 240336 chunks - 24/181 tiles por tiling
        # 9x6 - 540756 chunks - 54/181 tiles por tiling
        # 12x8 - 961344 chunks - 96/181 tiles por tiling
        # total - 1812534 chunks - 181/181 tiles por tiling
        for self.proj in self.proj_list:
            for self.tiling in self.tiling_list:
                try:
                    self.fitter = load_pickle(self.fitter_pickle_file)
                    continue
                except FileNotFoundError:
                    pass

                print(f'\tFitting - {self.metric} - {self.proj} - {self.tiling}...')
                samples = self.data_bucket[self.metric][self.proj][self.tiling]
                self.fitter = Fitter(samples, bins=self.bins, distributions=self.config['distributions'], timeout=1500)
                self.fitter.fit()
                print(f'\tSaving Fitter... ')
                save_pickle(self.fitter, self.fitter_pickle_file)

    def make_hist(self):
        if self.pdf_file.exists(): return
        for index2, self.proj in enumerate(self.proj_list):
            for index, self.tiling in enumerate(self.tiling_list, 1):
                print(f'  Make Histogram - {self.metric} {self.proj} {self.tiling} - {self.bins} bins')

                self.fitter = load_pickle(self.fitter_pickle_file)
                width = self.fitter.x[1] - self.fitter.x[0]
                cdf_bins_height = np.cumsum([y * width for y in self.fitter.y])
                error_key = 'sumsquare_error' if self.error_type == 'sse' else 'bic'
                error_sorted = self.fitter.df_errors[error_key].sort_values()[0:self.n_dist]
                dists = error_sorted.index
                fitted_pdf = self.fitter.fitted_pdf
                scilimits = self.metric_label[self.metric]['scilimits']
                xlabel = self.metric_label[self.metric]['xlabel']

                # <editor-fold desc="Make PDF">
                # Make bars of histogram
                ax: axes.Axes = self.fig_pdf.add_subplot(2, 5, index2 * 5 + index)
                ax.bar(self.fitter.x, self.fitter.y, label='empirical', color='#dbdbdb', width=width)

                # Make plot for n_dist distributions
                for dist_name in dists:
                    label = f'{dist_name} - {self.error_type.upper()} {error_sorted[dist_name]: 0.3e}'
                    ax.plot(self.fitter.x, fitted_pdf[dist_name], color=self.dists_colors[dist_name],
                            label=label)

                ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                ax.set_title(f'{self.proj.upper()} - {self.tiling}')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Density' if index in [1] else None)
                ax.legend(loc='upper right')
                # </editor-fold>

                # <editor-fold desc="Make CDF">
                # Make bars of CDF
                ax: axes.Axes = self.fig_cdf.add_subplot(2, 5, index2 * 5 + index)
                ax.bar(self.fitter.x, cdf_bins_height, label='empirical', color='#dbdbdb', width=width)

                # make plot for n_dist distributions cdf
                for dist_name in dists:
                    dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                    param = self.fitter.fitted_param[dist_name]
                    fitted_cdf = dist.cdf(self.fitter.x, *param)
                    label = f'{dist_name} - {self.error_type.upper()} {error_sorted[dist_name]: 0.3e}'
                    ax.plot(self.fitter.x, fitted_cdf, color=self.dists_colors[dist_name], label=label)

                ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                ax.set_title(f'{self.proj.upper()}-{self.tiling}')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Cumulative' if index in [1] else None)
                ax.legend(loc='lower right')
                # </editor-fold>

        print(f'  Saving the CDF and PDF ')
        self.fig_pdf.savefig(self.pdf_file)
        self.fig_cdf.savefig(self.cdf_file)

    def make_boxplot(self):
        if self.boxplot_file.exists(): return
        for index2, self.proj in enumerate(self.proj_list):
            for index, self.tiling in enumerate(self.tiling_list, 1):
                print(f'  make_boxplot - {self.metric} {self.proj} {self.tiling} - {self.bins} bins')

                self.data_bucket = load_json(self.data_bucket_file)
                samples = self.data_bucket[self.metric][self.proj][self.tiling]
                scilimits = self.metric_label[self.metric]['scilimits']

                ax: axes.Axes = self.fig_boxplot.add_subplot(2, 5, index2 * 5 + index)
                boxplot_sep = ax.boxplot((samples,), widths=0.8,
                                         # whis=(0, 100),
                                         showfliers=False,
                                         boxprops=dict(facecolor='tab:blue'),
                                         flierprops=dict(color='r'),
                                         medianprops=dict(color='k'),
                                         patch_artist=True)
                for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                ax.set_xticks([0])
                ax.set_xticklabels([self.tiling])
                ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                ax.title = self.tiling

        print(f'  Saving the figure')
        suptitle = self.metric_label[self.metric]['xlabel']
        self.fig_boxplot.suptitle(f'{suptitle}')
        self.fig_boxplot.savefig(self.boxplot_file)

    def calc_stats(self):
        if self.stats_file.exists():
            return
        self.clear_state()

        for self.metric in self.metric_list:
            self.data_bucket = load_json(self.data_bucket_file)
            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    print(f'[{self.metric}][{self.proj}][{self.tiling}] - Bins = {self.bins}')
                    samples = self.data_bucket[self.metric][self.proj][self.tiling]

                    # Calculate percentiles
                    percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

                    # Calculate errors
                    df_errors: pd.DataFrame = self.fitter.df_errors
                    sse: pd.Series = df_errors['sumsquare_error']
                    bic: pd.Series = df_errors['bic']
                    aic: pd.Series = df_errors['aic']
                    n_bins = len(self.fitter.x)
                    rmse = np.sqrt(sse / n_bins)
                    nrmse = rmse / (sse.max() - sse.min())

                    # Append info and stats on Dataframe
                    self.stats[f'proj'].append(self.proj)
                    self.stats[f'tiling'].append(self.tiling)
                    self.stats[f'metric'].append(self.metric)
                    self.stats[f'bins'].append(self.bins)

                    self.stats[f'average'].append(np.average(samples))
                    self.stats[f'std'].append(float(np.std(samples)))

                    self.stats[f'min'].append(percentile[0])
                    self.stats[f'quartile1'].append(percentile[1])
                    self.stats[f'median'].append(percentile[2])
                    self.stats[f'quartile3'].append(percentile[3])
                    self.stats[f'max'].append(percentile[4])

                    # Append distributions on Dataframe
                    for dist in sse.keys():
                        params = self.fitter.fitted_param[dist]
                        dist_info = self.find_dist(dist, params)

                        self.stats[f'rmse_{dist}'].append(rmse[dist])
                        self.stats[f'nrmse_{dist}'].append(nrmse[dist])
                        self.stats[f'sse_{dist}'].append(sse[dist])
                        self.stats[f'bic_{dist}'].append(bic[dist])
                        self.stats[f'aic_{dist}'].append(aic[dist])
                        self.stats[f'param_{dist}'].append(dist_info['parameters'])
                        self.stats[f'loc_{dist}'].append(dist_info['loc'])
                        self.stats[f'scale_{dist}'].append(dist_info['scale'])

        print(f'  Saving Stats')
        pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)

    def make_violinplot(self, overwrite=False):
        print(f'\n====== Make Violin - Bins = {self.bins} ======')
        folder = self.workfolder / 'violinplot'
        folder.mkdir(parents=True, exist_ok=True)

        subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
        colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
        data_bucket = load_json(self.data_bucket_file)
        legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                          # mpatches.Patch(color=colors['cmp'], label='CMP'),
                          ]

        # make an image for each metric and projection
        for mid, self.metric in enumerate(self.metric_list):
            for self.proj in self.proj_list:
                img_file = folder / f'violinplot_pattern_{mid}{self.metric}_{self.proj}.png'

                if img_file.exists() and not overwrite:
                    print(f'Figure exist. Skipping')
                    continue

                # <editor-fold desc="Format plot">
                if self.metric == 'time':
                    scilimits = (-3, -3)
                    title = f'Average Decoding {self.metric.capitalize()} (ms)'
                elif self.metric == 'time_std':
                    scilimits = (-3, -3)
                    title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                elif self.metric == 'rate':
                    scilimits = (6, 6)
                    title = f'Bit {self.metric.capitalize()} (Mbps)'
                else:
                    scilimits = (0, 0)
                    title = self.metric
                # </editor-fold>

                fig = figure.Figure(figsize=(6.8, 3.84))
                fig.suptitle(f'{title}')

                for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                    # Get data
                    tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                    if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                        tiling_data = [data for data in tiling_data if data < 1000]

                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.violinplot([tiling_data], positions=[1],
                                  showmedians=True, widths=0.9)

                    ax.set_xticks([1])
                    ax.set_xticklabels([self.tiling_list[index - 1]])
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                print(f'  Saving the figure')
                fig.savefig(img_file)

    def calc_corr(self):
        if self.correlations_file.exists():
            return

        # self.clear_state()
        self.metric_list.remove('time_std')
        print(f'  Processing Correlation')
        corretations_tree = AutoDict()
        for metric1, metric2 in combinations(self.metric_list, r=2):
            # if not (metric1 == 'rate' and metric2 == 'MSE'): continue
            for self.proj in self.proj_list:
                for n, self.name in enumerate(self.name_list, 1):
                    print(f'\t{metric1} x {metric2} - {self.proj} {self.name}.')
                    self.metric = metric1
                    video_data_metric_m1 = load_json(self.json_metrics)
                    self.metric = metric2
                    video_data_metric_m2 = load_json(self.json_metrics)

                    for self.tiling in self.tiling_list:
                        for self.quality in self.quality_list:
                            for self.tile in self.tile_list:
                                print(f'\r\t\t{self.tiling} {self.quality_str} {self.tile_str}.', end='')
                                tile_data1 = []
                                tile_data2 = []
                                for self.chunk in self.chunk_list:
                                    self.metric = metric1
                                    chunk_data1 = video_data_metric_m1[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                                    chunk_data2 = self.process_value(chunk_data1)
                                    tile_data1.append(chunk_data2)

                                    self.metric = metric2
                                    chunk_data1 = video_data_metric_m2[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                                    chunk_data2 = self.process_value(chunk_data1)
                                    tile_data2.append(chunk_data2)
                                corrcoef = np.corrcoef((tile_data1, tile_data2))[0][1]
                                if np.isnan(corrcoef):
                                    cov = np.cov((tile_data1, tile_data2))[0][1]
                                    var1 = np.cov((tile_data1, tile_data1))[0][1]
                                    var2 = np.cov((tile_data2, tile_data2))[0][1]
                                    prod = var1 * var2
                                    if prod == 0: prod = sys.float_info.min
                                    corrcoef = cov / prod
                                corretations_tree[self.proj][self.name][self.tiling][self.quality][self.tile][f'{metric1},{metric2}'] = corrcoef
        print(f'  Saving Correlation')
        save_pickle(corretations_tree, self.correlations_file)

    def calc_corr2(self):
        if self.correlations_file.exists():
            pass
            # return
        self.clear_state()
        self.metric_list.remove('time_std')
        len_metric = len(self.metric_list)
        self.tiling_list.remove('1x1')
        print(f'  Processing Correlation')
        corretations_tree = AutoDict()
        for self.proj in self.proj_list:
            for n, self.name in enumerate(self.name_list, 1):

                for id1 in range(len_metric-1):
                    metric1 = self.metric_list[id1]
                    self.metric = metric1
                    video_data_metric_m1 = load_json(self.json_metrics)
                    for id2 in range(id1+1, len_metric):
                        metric2 = self.metric_list[id2]
                        print(f'\t{metric1} x {metric2} - {self.proj} {self.name}.')
                        self.metric = metric2
                        video_data_metric_m2 = load_json(self.json_metrics)

                        for self.tiling in self.tiling_list:
                            for self.quality in self.quality_list:
                                avg1 = []
                                avg2 = []
                                for self.tile in self.tile_list:
                                    print(f'\r\t\t{self.tiling} {self.quality_str} {self.tile_str}.', end='')
                                    tile_data1 = []
                                    tile_data2 = []
                                    for self.chunk in self.chunk_list:
                                        chunk_data1 = video_data_metric_m1[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                                        chunk_data2 = self.process_value(chunk_data1, metric1)
                                        tile_data1.append(chunk_data2)

                                        chunk_data1 = video_data_metric_m2[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                                        chunk_data2 = self.process_value(chunk_data1, metric2)
                                        tile_data2.append(chunk_data2)

                                    avg1.append(np.average(tile_data1))
                                    avg2.append(np.average(tile_data2))
                                corrcoef = np.corrcoef((avg1, avg2))[0][1]
                                if np.isnan(corrcoef):
                                    cov = np.cov((avg1, avg2))[0][1]
                                    var1 = np.cov((avg1, avg1))[0][1]
                                    var2 = np.cov((avg2, avg2))[0][1]
                                    prod = var1 * var2
                                    if prod == 0: prod = sys.float_info.min
                                    corrcoef = cov / prod
                                corretations_tree[self.proj][self.name][self.tiling][self.quality][f'{metric1},{metric2}'] = corrcoef
                        print('')
        print(f'  Saving Correlation')
        save_pickle(corretations_tree, self.correlations_file.with_suffix('.pickle'))


class ByPatternByQualityProps(ByPattern):

    @property
    def fitter_pickle_file(self) -> Path:
        """
        Need: metric, proj, tiling, and bins

        :return:  Path(fitter_pickle_file)
        """
        fitter_file = self.workfolder_fitter / f'fitter_{self.metric}_{self.proj}_{self.tiling}_{self.quality}_{self.bins}bins.pickle'
        return fitter_file

    @property
    def cdf_file2(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        return self.workfolder_pdf_cdf / f'cdf_{self.metric}_{self.quality_str}.png'

    @property
    def cdf_file1(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        return self.workfolder_pdf_cdf / f'cdf_{self.metric}_{self.tiling}.png'

    @property
    def pdf_file2(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        return self.workfolder_pdf_cdf / f'pdf_{self.metric}_{self.quality_str}.png'

    @property
    def pdf_file1(self) -> Path:
        """
        Need: proj, and metric
        :return:
        """
        return self.workfolder_pdf_cdf / f'pdf_{self.metric}_{self.tiling}.png'


class ByPatternByQuality(ByPatternByQualityProps):
    def main(self):
        print(f'\n====== ByPatternByQuality - error_type={self.error_type}, n_dist={self.n_dist} ======')
        for self.metric in self.metric_list:
            # self.get_data_bucket()
            # self.make_fit()
            # self.make_hist1()  # compare tiling
            self.make_hist2()  # compare tiling
            # self.calc_stats()
            # self.make_boxplot()

        # self.calc_corr()

    def get_data_bucket(self):
        tiles_num = {"1x1": 1680,
                     "3x2": 10080,
                     "6x4": 40320,
                     "9x6": 90720,
                     "12x8": 161280}

        try:
            self.data_bucket = load_json(self.data_bucket_file)
            return
        except FileNotFoundError:
            self.data_bucket = AutoDict()

        for self.video in self.video_list:
            video_data_metric = load_json(self.json_metrics)
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    print(f'\r\t{self.metric} {self.video} {self.tiling} {self.quality}')
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            chunk_data1 = video_data_metric[self.proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                            chunk_data2 = self.process_value(chunk_data1)
                            try:
                                self.data_bucket[self.metric][self.proj][self.tiling][self.quality].append(chunk_data2)
                            except AttributeError:
                                self.data_bucket[self.metric][self.proj][self.tiling][self.quality] = [chunk_data2]

        # test
        for proj in self.proj_list:
            for tiling in self.tiling_list:
                for quality in self.quality_list:
                    bucket = self.data_bucket[self.metric][proj][tiling][quality]
                    if tiles_num[tiling] != len(bucket):
                        self.log(f'bucket size error', self.data_bucket_file)
                        print(f'{Bcolors.RED}\n    bucket size error')

        print(f'  Saving ... ', end='')
        save_json(self.data_bucket, self.data_bucket_file)
        print(f'  Finished.')

        self.video = self.tiling = self.quality = self.tile = self.chunk = None

    def make_fit(self):
        # "deve salvar o arquivo"
        #                                56  x    6   x  var  x 60
        # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
        # 1x1 - 10014 chunks - 1/181 tiles por tiling
        # 3x2 - 60084 chunks - 6/181 tiles por tiling
        # 6x4 - 240336 chunks - 24/181 tiles por tiling
        # 9x6 - 540756 chunks - 54/181 tiles por tiling
        # 12x8 - 961344 chunks - 96/181 tiles por tiling
        # total - 1812534 chunks - 181/181 tiles por tiling
        for self.proj in self.proj_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    try:
                        self.fitter = load_pickle(self.fitter_pickle_file)
                        continue
                    except FileNotFoundError:
                        pass

                    print(f'\tFitting - {self.metric} - {self.proj} - {self.tiling} - {self.quality_str}... ')

                    samples = self.data_bucket[self.metric][self.proj][self.tiling][self.quality]
                    self.fitter = Fitter(samples, bins=self.bins, distributions=self.config['distributions'], timeout=1500)
                    self.fitter.fit()
                    print(f'\tSaving Fitter... ')
                    save_pickle(self.fitter, self.fitter_pickle_file)
        self.proj = self.tiling = self.quality = None

    _fig_pdf: plt.Figure
    _fig_cdf: plt.Figure

    def make_hist1(self):
        for self.tiling in self.tiling_list:
            if self.pdf_file.exists(): return
            idx = 0
            self._fig_pdf = plt.figure(figsize=(15.0, 5),
                                       dpi=600,
                                       linewidth=0.5,
                                       tight_layout=True)
            self._fig_cdf = plt.figure(figsize=(15.0, 5),
                                       dpi=600,
                                       linewidth=0.5,
                                       tight_layout=True)
            for self.proj in self.proj_list:
                for self.quality in self.quality_list:
                    idx += 1
                    cols = len(self.quality_list)
                    self.hist_base(idx, cols)

            print(f'  Saving the CDF and PDF ')
            self._fig_pdf.savefig(self.pdf_file1)
            self._fig_cdf.savefig(self.cdf_file1)

        self.proj = self.tiling = self.quality = None

    def make_hist2(self):
        for self.quality in self.quality_list:
            # if self.pdf_file.exists(): return
            idx = 0
            self._fig_pdf = plt.figure(figsize=(12.0, 5),
                                       dpi=600,
                                       linewidth=0.5,
                                       tight_layout=True)
            self._fig_cdf = plt.figure(figsize=(12.0, 5),
                                       dpi=600,
                                       linewidth=0.5,
                                       tight_layout=True)
            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    idx += 1
                    cols = len(self.tiling_list)
                    self.hist_base(idx, cols)

            print(f'  Saving the CDF and PDF ')
            self._fig_pdf.savefig(self.pdf_file2)
            self._fig_cdf.savefig(self.cdf_file2)

    def hist_base(self, idx, cols):
        print(f'    Make Histogram - {self.metric} {self.proj} {self.tiling}  {self.quality_str} - {self.bins} bins')

        self.fitter = load_pickle(self.fitter_pickle_file)
        width = self.fitter.x[1] - self.fitter.x[0]
        cdf_bins_height = np.cumsum([y * width for y in self.fitter.y])
        error_key = 'sumsquare_error' if self.error_type == 'sse' else 'bic'
        error_sorted = self.fitter.df_errors[error_key].sort_values()[0:self.n_dist]
        dists = error_sorted.index
        fitted_pdf = self.fitter.fitted_pdf
        scilimits = self.metric_label[self.metric]['scilimits']
        xlabel = self.metric_label[self.metric]['xlabel']

        # <editor-fold desc="Make PDF">
        # Make bars of histogram
        ax: axes.Axes = self._fig_pdf.add_subplot(2, cols, idx)
        ax.bar(self.fitter.x, self.fitter.y, label='empirical', color='#dbdbdb', width=width)

        # Make plot for n_dist distributions
        for dist_name in dists:
            label = f'{dist_name} - {self.error_type.upper()} {error_sorted[dist_name]: 0.3e}'
            ax.plot(self.fitter.x, fitted_pdf[dist_name], color=self.dists_colors[dist_name],
                    label=label)

        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_title(f'{self.proj.upper()} - {self.tiling} - {self.quality_str}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density' if idx in [1] else None)
        ax.legend(loc='upper right')

        # </editor-fold>

        # <editor-fold desc="Make CDF">
        # Make bars of CDF
        ax: axes.Axes = self._fig_cdf.add_subplot(2, cols, idx)
        ax.bar(self.fitter.x, cdf_bins_height, label='empirical', color='#dbdbdb', width=width)

        # make plot for n_dist distributions cdf
        for dist_name in dists:
            dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
            param = self.fitter.fitted_param[dist_name]
            fitted_cdf = dist.cdf(self.fitter.x, *param)
            label = f'{dist_name} - {self.error_type.upper()} {error_sorted[dist_name]: 0.3e}'
            ax.plot(self.fitter.x, fitted_cdf, color=self.dists_colors[dist_name], label=label)

        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_title(f'{self.proj.upper()}-{self.tiling}-{self.quality_str}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative' if idx in [1] else None)
        ax.legend(loc='lower right')
        # </editor-fold>

    def calc_stats(self):
        if self.stats_file.exists():
            return

        print(f'{self.state_str()}[{self.metric}] - Bins = {self.bins}')
        samples = self.data_bucket[self.metric][self.proj][self.tiling]

        # Calculate percentiles
        percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

        # Calculate errors
        df_errors: pd.DataFrame = self.fitter.df_errors
        sse: pd.Series = df_errors['sumsquare_error']
        bic: pd.Series = df_errors['bic']
        aic: pd.Series = df_errors['aic']
        n_bins = len(self.fitter.x)
        rmse = np.sqrt(sse / n_bins)
        nrmse = rmse / (sse.max() - sse.min())

        # Append info and stats on Dataframe
        self.stats[f'proj'].append(self.proj)
        self.stats[f'tiling'].append(self.tiling)
        self.stats[f'metric'].append(self.metric)
        self.stats[f'bins'].append(self.bins)

        self.stats[f'average'].append(np.average(samples))
        self.stats[f'std'].append(float(np.std(samples)))

        self.stats[f'min'].append(percentile[0])
        self.stats[f'quartile1'].append(percentile[1])
        self.stats[f'median'].append(percentile[2])
        self.stats[f'quartile3'].append(percentile[3])
        self.stats[f'max'].append(percentile[4])

        # Append distributions on Dataframe
        for dist in sse.keys():
            params = self.fitter.fitted_param[dist]
            dist_info = self.find_dist(dist, params)

            self.stats[f'rmse_{dist}'].append(rmse[dist])
            self.stats[f'nrmse_{dist}'].append(nrmse[dist])
            self.stats[f'sse_{dist}'].append(sse[dist])
            self.stats[f'bic_{dist}'].append(bic[dist])
            self.stats[f'aic_{dist}'].append(aic[dist])
            self.stats[f'param_{dist}'].append(dist_info['parameters'])
            self.stats[f'loc_{dist}'].append(dist_info['loc'])
            self.stats[f'scale_{dist}'].append(dist_info['scale'])

        if self.metric == self.metric_list[-1] and self.tiling == self.tiling_list[-1]:
            print(f'  Saving Stats')
            pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)

    def calc_corr(self):
        if self.correlations_file.exists(): return

        print(f'  Processing Correlation')
        corretations_bucket = defaultdict(list)
        metric_list = ['rate', 'time']
        for metric1, metric2 in combinations(metric_list, r=2):
            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    print(f'{self.state_str()}')
                    corrcoef = []
                    for self.name in self.name_list:
                        self.metric = metric1
                        result_m1 = load_json(self.json_metrics)
                        self.metric = metric2
                        result_m2 = load_json(self.json_metrics)

                        for self.quality in self.quality_list:
                            for self.tile in self.tile_list:
                                self.metric = metric1
                                self.results = result_m1
                                tile_data1 = [np.average(self.chunk_results) for self.chunk in self.chunk_list]
                                if self.metric not in ['time', 'rate']:
                                    tile_data1 = [data[self.metric] for data in tile_data1]

                                self.metric = metric2
                                self.results = result_m2
                                tile_data2 = [np.average(self.chunk_results) for self.chunk in self.chunk_list]
                                if self.metric not in ['time', 'rate']:
                                    tile_data2 = [data[self.metric] for data in tile_data2]

                                corrcoef += [np.corrcoef((tile_data1, tile_data2))[1][0]]

                    corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                    corretations_bucket[f'proj'].append(self.proj)
                    corretations_bucket[f'tiling'].append(self.tiling)
                    corretations_bucket[f'corr'].append(np.average(corrcoef))
        print(f'  Saving Correlation')
        pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)

    def make_boxplot(self):
        # todo: projetar esse gráfico
        if self.boxplot_file.exists(): return
        for index2, self.proj in enumerate(self.proj_list):
            for index, self.tiling in enumerate(self.tiling_list, 1):
                for index0, self.quality in enumerate(self.quality_list, 1):
                    print(f'  make_boxplot - {self.metric} {self.proj} {self.tiling} {self.quality} - {self.bins} bins')
                    self.data_bucket = load_json(self.data_bucket_file)
                    samples = self.data_bucket[self.metric][self.proj][self.tiling][self.quality]

                    scilimits = self.metric_label[self.metric]['scilimits']

                    ax: axes.Axes = self.fig_boxplot.add_subplot(2, 5, index2 * 5 + index)
                    boxplot_sep = ax.boxplot((samples,), widths=0.8,
                                             whis=(0, 100),
                                             showfliers=False,
                                             boxprops=dict(facecolor='tab:blue'),
                                             flierprops=dict(color='r'),
                                             medianprops=dict(color='k'),
                                             patch_artist=True)
                    for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                    ax.set_xticks([0])
                    ax.set_xticklabels([self.tiling])
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                    if self.tiling == self.tiling_list[-1]:
                        print(f'  Saving the figure')
                        suptitle = self.metric_label[self.metric]['xlabel']
                        self.fig_boxplot.suptitle(f'{suptitle}')
                        self.fig_boxplot.savefig(self.boxplot_file)

    def make_violinplot(self, overwrite=False):
        print(f'\n====== Make Violin - Bins = {self.bins} ======')
        folder = self.workfolder / 'violinplot'
        folder.mkdir(parents=True, exist_ok=True)

        subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
        colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
        data_bucket = load_json(self.data_bucket_file)
        legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                          # mpatches.Patch(color=colors['cmp'], label='CMP'),
                          ]

        # make an image for each metric and projection
        for mid, self.metric in enumerate(self.metric_list):
            for self.proj in self.proj_list:
                img_file = folder / f'violinplot_pattern_{mid}{self.metric}_{self.proj}.png'

                if img_file.exists() and not overwrite:
                    print(f'Figure exist. Skipping')
                    continue

                # <editor-fold desc="Format plot">
                if self.metric == 'time':
                    scilimits = (-3, -3)
                    title = f'Average Decoding {self.metric.capitalize()} (ms)'
                elif self.metric == 'time_std':
                    scilimits = (-3, -3)
                    title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                elif self.metric == 'rate':
                    scilimits = (6, 6)
                    title = f'Bit {self.metric.capitalize()} (Mbps)'
                else:
                    scilimits = (0, 0)
                    title = self.metric
                # </editor-fold>

                fig = figure.Figure(figsize=(6.8, 3.84))
                fig.suptitle(f'{title}')

                for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                    # Get data
                    tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                    if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                        tiling_data = [data for data in tiling_data if data < 1000]

                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.violinplot([tiling_data], positions=[1],
                                  showmedians=True, widths=0.9)

                    ax.set_xticks([1])
                    ax.set_xticklabels([self.tiling_list[index - 1]])
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                print(f'  Saving the figure')
                fig.savefig(img_file)


DectimeGraphsOptions = {'0': ByPatternMakeBucket,
                        '1': ByPatternByQuality,
                        '2': ByPatternFit,
                        '3': ByPattern,
                        '4': ByPattern,
                        '5': ByPattern,
                        }
