import os
from pathlib import Path

from config.config import Config
from lib.assets.context import Context
from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.util import run_command


class MakeDecodable(Worker, MakeDecodablePaths):
    quality_list: list[str] = None
    decode_check = False

    def init(self):
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self):
                self.work()

    def work(self):
        self.assert_decodable()
        self.assert_dash()
        self.make_decodable_cmd()
        self.run()

    def assert_decodable(self):
        try:
            self._check_decodable()
            raise AbortError(f'decodable_chunk is OK.')
        except FileNotFoundError:
            pass

    def assert_dash(self):
        msg = []
        if not self.dash_m4s.exists():
            msg += ['dash_m4s not exist']
            self.logger.register_log('Dash M4S not found.', self.dash_m4s)
        if not self.dash_init.exists():
            msg += ['dash_init not exist']
            self.logger.register_log('Dash init not found.', self.dash_m4s)
        if msg:
            raise AbortError('/'.join(msg))

    cmd: str

    def make_decodable_cmd(self):
        self.cmd = (f'bash -c "cat {self.dash_init.as_posix()} {self.dash_m4s.as_posix()} '
                    f'> {self.decodable_chunk.as_posix()}"')

    def run(self):
        run_command(self.cmd, folder=self.decodable_folder,
                    ui_prefix='\t')

    def _check_decodable(self):
        chunk_size = self.decodable_chunk.stat().st_size
        if chunk_size == 0:
            self.logger.register_log('Chunk size is 0.', self.decodable_chunk)
            self.decodable_chunk.unlink()
            raise FileNotFoundError()


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

    config_file = Path('config/config_erp_qp.json')
    # config_file = Path('config/config_cmp_qp.json')
    videos_file = Path('config/videos_reduced.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MakeDecodable(ctx)
