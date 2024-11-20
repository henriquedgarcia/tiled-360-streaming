import json
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.makesitipaths import MakeSitiPaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.siti import SiTi
from lib.utils.worker_utils import load_json, save_json


def save_siti(results, results_file):
    if not results_file.parent.exists(): results_file.parent.mkdir(exist_ok=True, parents=True)
    results_file.write_text(json.dumps(results, indent=2))


class MakeSiti(Worker, CtxInterface):
    siti: SiTi = None
    make_siti_paths: MakeSitiPaths = None
    siti_results_df: pd.DataFrame = None

    def main(self):
        self.init()
        for self.name in self.name_list:
            with task(self):
                self.calc_siti()

        for self.name in self.name_list:
            with task(self):
                self.calc_stats()

        for self.name in self.name_list:
            with task(self):
                self.scatter_plot_siti()
                self.plot_siti()

    def init(self):
        self.make_siti_paths = MakeSitiPaths(self.ctx)
        self.projection = 'cmp'
        self.tiling = '1x1'
        self.tile = '0'
        self.quality = '0'

    def calc_siti(self):
        if self.siti_results.exists():
            new_name = self.siti_stats.with_suffix('.json')
            self.siti_stats.rename(new_name)
            return

        if not self.tile_video.exists():
            self.logger.register_log('compressed_file NOT_FOUND', self.tile_video)
            raise AbortError(f'compressed_file not exist. Skipping.')

        siti = SiTi(self.tile_video)
        save_json(siti.siti, self.siti_results)

    def calc_stats(self):
        if self.siti_stats.exists():
            print(f'{self.siti_stats} - the file exist')
            return

        siti_results = load_json(self.siti_results)

        si = siti_results['si']
        ti = siti_results['ti']
        bitrate = self.tile_video.stat().st_size * 8 / 60

        siti_stats = defaultdict(list)
        siti_stats['group'].append(self.group)
        siti_stats['proj'].append(self.projection)
        siti_stats['video'].append(self.name)
        siti_stats['name'].append(self.name)
        siti_stats['tiling'].append(self.tiling)
        siti_stats['tile'].append(self.tile)
        siti_stats['quality'].append(self.quality)
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

        if self.name == self.name_list[-1]:
            pd.DataFrame(siti_stats).to_csv(self.siti_stats, index=False)

    def plot_siti(self):
        def plot1():
            from typing import Optional
            ax1: Optional[plt.Axes] = None
            ax2: Optional[plt.Axes] = None
            fig = plt.Figure()
            if self.name == self.name_list[0]:
                fig, (ax1, ax2) = plt.subplots(2,
                                               1,
                                               figsize=(8, 6),
                                               dpi=300)

            siti_results = load_json(self.siti_results)
            name = self.name.replace('_nas',
                                     '')
            si = siti_results[self.name]['si']
            ti = siti_results[self.name]['ti']
            ax1.plot(si,
                     label=name)
            ax2.plot(ti,
                     label=name)

            if self.name == self.name_list[-1]:
                ax1.set_xlabel("Frame")
                ax1.set_ylabel("Spatial Information")

                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Temporal Information')

                handles, labels = ax1.get_legend_handles_labels()
                fig.suptitle('SI/TI by frame')
                fig.legend(handles,
                           labels,
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
                fig, (ax1, ax2) = plt.subplots(2,
                                               1,
                                               figsize=(8, 6),
                                               dpi=300)
                for proj in proj_list:
                    self.video = name.replace('_nas',
                                              f'_{proj}_nas')

                    siti_results_df = pd.read_csv(self.siti_results)
                    si = siti_results_df['si']
                    ti = siti_results_df['ti']
                    ax1.plot(si,
                             label=self.video)
                    ax2.plot(ti,
                             label=self.video)

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

        def change_name(x):
            return x.replace('_nas',
                             '')

        siti_stats['video'].apply(change_name)

        si_max = siti_stats['si_med'].max()
        ti_max = siti_stats['ti_med'].max()

        siti_erp = siti_stats['proj'] == 'erp'
        siti_stats_erp = siti_stats[siti_erp][['video', 'si_med', 'ti_med']]
        fig_erp, ax_erp = plt.subplots(1,
                                       1,
                                       figsize=(8, 6),
                                       dpi=300)

        for idx, (video, si, ti) in siti_stats_erp.iterrows():
            ax_erp.scatter(si,
                           ti,
                           label=video + ' ')

        ax_erp.set_xlabel("Spatial Information")
        ax_erp.set_ylabel("Temporal Information")
        ax_erp.set_xlim(xmax=si_max + 5,
                        xmin=0)
        ax_erp.set_ylim(ymax=ti_max + 5,
                        ymin=0)
        ax_erp.legend(loc='upper left',
                      bbox_to_anchor=(1.01, 1.0),
                      fontsize='small')

        fig_erp.suptitle('ERP - SI x TI')
        fig_erp.tight_layout()
        fig_erp.show()
        fig_erp.savefig(self.siti_folder / 'scatter_ERP.png')

        ############################################
        siti_cmp = siti_stats['proj'] == 'cmp'
        siti_stats_cmp = siti_stats[siti_cmp][['video', 'si_med', 'ti_med']]
        fig_cmp, ax_cmp = plt.subplots(1,
                                       1,
                                       figsize=(8, 6),
                                       dpi=300)

        for idx, (video, si, ti) in siti_stats_cmp.iterrows():
            ax_cmp.scatter(si,
                           ti,
                           label=video)

        ax_cmp.set_xlabel("Spatial Information")
        ax_cmp.set_ylabel("Temporal Information")
        ax_cmp.set_xlim(xmax=si_max + 5,
                        xmin=0)
        ax_cmp.set_ylim(ymax=ti_max + 5,
                        ymin=0)
        ax_cmp.legend(loc='upper left',
                      bbox_to_anchor=(1.01, 1.0),
                      fontsize='small')

        fig_cmp.suptitle('CMP - SI x TI')
        fig_cmp.tight_layout()
        fig_cmp.show()
        fig_cmp.savefig(self.siti_folder / 'scatter_CMP.png')

    @property
    def siti_plot(self):
        return self.make_siti_paths.siti_plot

    @property
    def siti_folder(self):
        return self.make_siti_paths.siti_folder

    @property
    def siti_results(self):
        return self.make_siti_paths.siti_results

    @property
    def siti_stats(self):
        return self.make_siti_paths.siti_stats

    @property
    def tile_video(self):
        return self.make_siti_paths.make_tiles_paths.tile_video
