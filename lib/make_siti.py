from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lib.assets.worker import Worker
from lib.utils.siti import SiTi
from lib.utils.util import load_json


class MakeSiti(Worker):
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
                self.log('compressed_file NOT_FOUND',
                         self.compressed_file)
                print(f'compressed_file not exist {self.compressed_file}. Skipping.')
                continue

            siti = SiTi(self.compressed_file)

            siti_results_df = pd.DataFrame(siti.siti)
            siti_results_df.to_csv(self.siti_results)

    def calc_stats(self):
        siti_stats = defaultdict(list)
        if self.siti_stats.exists():
            print(f'{self.siti_stats} - the file exist')

            # def calc():
            #     siti_stats = pd.read_csv(self.siti_stats)
            #     siti_stats1 = siti_stats[['group', 'name', 'proj', 'si_med', 'ti_med', 'bitrate']]
            #     # siti_stats2 = siti_stats1.sort_values('name').sort_values('proj').sort_values('group')
            #     mid_x = pd.MultiIndex.from_frame(siti_stats1[['group', 'name', 'proj']])
            #     data = siti_stats1[['si_med', 'ti_med', 'bitrate']]
            #     siti_stats3 = pd.DataFrame(data.values, index=mid_x)

            return

        for self.video in self.video_list:
            siti_results = pd.read_csv(self.siti_results,
                                       index_col=0)
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

        pd.DataFrame(siti_stats).to_csv(self.siti_stats,
                                        index=False)

    def plot_siti(self):
        def plot1():
            fig, (ax1, ax2) = plt.subplots(2,
                                           1,
                                           figsize=(8, 6),
                                           dpi=300)
            for self.video in self.video_list:
                siti_results = load_json(self.siti_results)
                name = self.name.replace('_nas',
                                         '')
                si = siti_results[self.video]['si']
                ti = siti_results[self.video]['ti']
                ax1.plot(si,
                         label=name)
                ax2.plot(ti,
                         label=name)

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
        fig_erp.savefig(self.project_path / self.siti_folder / 'scatter_ERP.png')

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
        fig_cmp.savefig(self.project_path / self.siti_folder / 'scatter_CMP.png')
