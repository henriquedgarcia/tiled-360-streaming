import os
from pathlib import Path

import pandas as pd

from config.config import Config
from lib.assets.context import Context
from lib.assets.paths.makesitipaths import MakeSitiPaths
from lib.assets.paths.maketilespaths import MakeTilesPaths
from lib.assets.siti import SiTi
from lib.assets.worker import Worker
from lib.utils.util import print_error


class MakeSiti(Worker, MakeSitiPaths, MakeTilesPaths):
    siti: SiTi = None
    siti_results_df: pd.DataFrame = None

    def init(self):
        self.tiling = '3x2'
        self.quality = '28'

    def main(self):
        def assert_siti_csv_results() -> bool:
            try:
                csv_results = pd.read_csv(self.siti_csv_results, index_col=0)
                if csv_results.size == 0:
                    self.siti_csv_results.unlink()
                    raise FileNotFoundError
                print_error(f'\t{self.name} siti_csv_results exist.')
                return True
            except FileNotFoundError:
                return False

        def assert_tile_video() -> bool:
            if self.tile_video.exists():
                return True

            self.logger.register_log('compressed_file NOT_FOUND', self.tile_video)
            print_error('\tcompressed_file not exist. Skipping.')
            return False

        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tile in self.tile_list:
                    print(f'{self.name}_{self.tile}')
                    if assert_siti_csv_results(): continue
                    if not assert_tile_video(): continue

                    siti = SiTi(self.tile_video)
                    df = pd.DataFrame(siti.siti)
                    df.to_csv(self.siti_csv_results, index_label='frame')

        # siti_stats = defaultdict(list)
        # self.dict = defaultdict(list)
        # for self.name in self.name_list:
        #     for self.tile in self.tile_list:
        #         with task(self):
        #             self.calc_stats(siti_stats)
        # pd.DataFrame(siti_stats).to_csv(self.siti_stats, index=False)
        # new_df = pd.DataFrame(self.dict)
        # new_df.set_index(['name', 'projection', 'tiling', 'tile', 'quality', 'frame'], inplace=True)
        # new_df.to_pickle(self.siti_result_pickle)
        # for self.name in self.name_list:
        #     with task(self):
        #         self.calc_stats()

        # for self.name in self.name_list:
        #     with task(self):
        #         self.scatter_plot_siti()
        #         self.plot_siti()

    # def calc_stats(self, siti_stats):
    #     # if self.siti_stats.exists():
    #     #     print(f'{self.siti_stats} - the file exist')
    #     #     return
    #
    #     siti_results = pd.read_csv(self.siti_csv_results, index_col=0)
    #
    #     si = siti_results['si']
    #     ti = siti_results['ti']
    #     for frame, (si_, ti_) in enumerate(zip(si, ti)):
    #         self.dict['name'].append(self.name)
    #         self.dict['projection'].append(self.projection)
    #         self.dict['tiling'].append(self.tiling)
    #         self.dict['tile'].append(self.tile)
    #         self.dict['quality'].append(self.quality)
    #         self.dict['frame'].append(frame)
    #         self.dict['si'].append(si_)
    #         self.dict['ti'].append(ti_)
    #     bitrate = self.tile_video.stat().st_size * 8 / 60
    #
    #     siti_stats['group'].append(self.group)
    #     siti_stats['name'].append(self.name)
    #     siti_stats['proj'].append(self.projection)
    #     siti_stats['tiling'].append(self.tiling)
    #     siti_stats['tile'].append(self.tile)
    #     siti_stats['quality'].append(self.quality)
    #     siti_stats['bitrate'].append(bitrate)
    #
    #     siti_stats['si_avg'].append(np.average(si))
    #     siti_stats['si_std'].append(np.std(si))
    #     siti_stats['si_max'].append(np.max(si))
    #     siti_stats['si_min'].append(np.min(si))
    #     siti_stats['si_med'].append(np.median(si))
    #     siti_stats['ti_avg'].append(np.average(ti))
    #     siti_stats['ti_std'].append(np.std(ti))
    #     siti_stats['ti_max'].append(np.max(ti))
    #     siti_stats['ti_min'].append(np.min(ti))
    #     siti_stats['ti_med'].append(np.median(ti))
    #     siti_stats['corr'].append(np.corrcoef(si, ti)[0, 1])

    # def plot_siti(self):
    #     def plot1():
    #         from typing import Optional
    #         ax1: Optional[plt.Axes] = None
    #         ax2: Optional[plt.Axes] = None
    #         fig = plt.Figure()
    #         if self.name == self.name_list[0]:
    #             fig, (ax1, ax2) = plt.subplots(2,
    #                                            1,
    #                                            figsize=(8, 6),
    #                                            dpi=300)
    #
    #         siti_results = load_json(self.siti_csv_results)
    #         name = self.name.replace('_nas',
    #                                  '')
    #         si = siti_results[self.name]['si']
    #         ti = siti_results[self.name]['ti']
    #         ax1.plot(si,
    #                  label=name)
    #         ax2.plot(ti,
    #                  label=name)
    #
    #         if self.name == self.name_list[-1]:
    #             ax1.set_xlabel("Frame")
    #             ax1.set_ylabel("Spatial Information")
    #
    #             ax2.set_xlabel('Frame')
    #             ax2.set_ylabel('Temporal Information')
    #
    #             handles, labels = ax1.get_legend_handles_labels()
    #             fig.suptitle('SI/TI by frame')
    #             fig.legend(handles,
    #                        labels,
    #                        loc='upper left',
    #                        bbox_to_anchor=[0.8, 0.93],
    #                        fontsize='small')
    #             fig.tight_layout()
    #             fig.subplots_adjust(right=0.78)
    #             fig.savefig(self.siti_all_plot)
    #             fig.show()
    #
    #     def plot2():
    #         proj_list = ['erp', 'cmp']
    #         for name in self.name_list:
    #             fig, (ax1, ax2) = plt.subplots(2,
    #                                            1,
    #                                            figsize=(8, 6),
    #                                            dpi=300)
    #             for proj in proj_list:
    #                 self.video = name.replace('_nas',
    #                                           f'_{proj}_nas')
    #
    #                 siti_results_df = pd.read_csv(self.siti_csv_results)
    #                 si = siti_results_df['si']
    #                 ti = siti_results_df['ti']
    #                 ax1.plot(si,
    #                          label=self.video)
    #                 ax2.plot(ti,
    #                          label=self.video)
    #
    #             ax1.set_xlabel("Frame")
    #             ax1.set_ylabel("Spatial Information")
    #
    #             ax2.set_xlabel('Frame')
    #             ax2.set_ylabel('Temporal Information')
    #
    #             fig.suptitle(f'SI/TI by frame - {name}')
    #             fig.legend(loc='upper left',
    #                        bbox_to_anchor=[0.8, 0.93],
    #                        fontsize='small')
    #             fig.tight_layout()
    #             fig.subplots_adjust(right=0.78)
    #             fig.savefig(self.siti_folder / f'siti_plot_{name}.png')
    #             fig.show()
    #
    #     plot1()
    #     plot2()

    # def scatter_plot_siti(self):
    #     siti_stats = pd.read_csv(self.siti_stats)
    #
    #     def change_name(x):
    #         return x.replace('_nas',
    #                          '')
    #
    #     siti_stats['video'].apply(change_name)
    #
    #     si_max = siti_stats['si_med'].max()
    #     ti_max = siti_stats['ti_med'].max()
    #
    #     siti_erp = siti_stats['proj'] == 'erp'
    #     siti_stats_erp = siti_stats[siti_erp][['video', 'si_med', 'ti_med']]
    #     fig_erp, ax_erp = plt.subplots(1,
    #                                    1,
    #                                    figsize=(8, 6),
    #                                    dpi=300)
    #
    #     for idx, (video, si, ti) in siti_stats_erp.iterrows():
    #         ax_erp.scatter(si,
    #                        ti,
    #                        label=video + ' ')
    #
    #     ax_erp.set_xlabel("Spatial Information")
    #     ax_erp.set_ylabel("Temporal Information")
    #     ax_erp.set_xlim(xmax=si_max + 5,
    #                     xmin=0)
    #     ax_erp.set_ylim(ymax=ti_max + 5,
    #                     ymin=0)
    #     ax_erp.legend(loc='upper left',
    #                   bbox_to_anchor=(1.01, 1.0),
    #                   fontsize='small')
    #
    #     fig_erp.suptitle('ERP - SI x TI')
    #     fig_erp.tight_layout()
    #     fig_erp.show()
    #     fig_erp.savefig(self.siti_folder / 'scatter_ERP.png')
    #
    #     ############################################
    #     siti_cmp = siti_stats['proj'] == 'cmp'
    #     siti_stats_cmp = siti_stats[siti_cmp][['video', 'si_med', 'ti_med']]
    #     fig_cmp, ax_cmp = plt.subplots(1,
    #                                    1,
    #                                    figsize=(8, 6),
    #                                    dpi=300)
    #
    #     for idx, (video, si, ti) in siti_stats_cmp.iterrows():
    #         ax_cmp.scatter(si,
    #                        ti,
    #                        label=video)
    #
    #     ax_cmp.set_xlabel("Spatial Information")
    #     ax_cmp.set_ylabel("Temporal Information")
    #     ax_cmp.set_xlim(xmax=si_max + 5,
    #                     xmin=0)
    #     ax_cmp.set_ylim(ymax=ti_max + 5,
    #                     ymin=0)
    #     ax_cmp.legend(loc='upper left',
    #                   bbox_to_anchor=(1.01, 1.0),
    #                   fontsize='small')
    #
    #     fig_cmp.suptitle('CMP - SI x TI')
    #     fig_cmp.tight_layout()
    #     fig_cmp.show()
    #     fig_cmp.savefig(self.siti_folder / 'scatter_CMP.png')


class GetMakeSiti(MakeSiti):
    def main(self):
        self.tiling = '3x2'
        self.quality = '28'
        result = []
        typos = {'name': str, 'projection': str, 'tiling': str, 'tile': int, 'quality': int, 'frame': int, 'si': list[float], 'ti': list[float]}
        keys = tuple(typos.keys())

        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tile in self.tile_list:
                    if not self.siti_csv_results.exists():
                        print_error(f'\t{self.name} siti_csv_results NOT exist. Skipping.')
                        continue

                    df = pd.read_csv(self.siti_csv_results, index_col=0)
                    si = df['si'].to_list()
                    ti = df['ti'].to_list()
                    for frame, (si, ti) in enumerate(zip(si, ti)):
                        result.append([self.name, self.projection, self.tiling, int(self.tile), int(self.quality), frame, si, ti])

        df = pd.DataFrame(result, columns=keys)
        df.set_index(keys[:-2], inplace=True)
        if df.size == 172800:
            df.to_pickle(self.siti_pickle_results)


if __name__ == '__main__':
    os.chdir('../')

    # config_file = 'config_erp_qp.json'
    # config_file = 'config_cmp_crf.json'
    # config_file = 'config_erp_crf.json'
    # videos_file = 'videos_reversed.json'
    # videos_file = 'videos_lumine.json'
    # videos_file = 'videos_container0.json'
    # videos_file = 'videos_container1.json'
    # videos_file = 'videos_fortrek.json'
    # videos_file = 'videos_hp_elite.json'
    # videos_file = 'videos_alambique.json'
    # videos_file = 'videos_test.json'
    # videos_file = 'videos_full.json'

    # config_file = Path('config/config_erp_qp.json')
    config_file = Path('config/config_cmp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    # MakeSiti(ctx)
    GetMakeSiti(ctx)
