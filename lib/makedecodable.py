from lib.assets.ctxinterface import CtxInterface
from lib.assets.errors import AbortError
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.worker import Worker
from lib.utils.context_utils import task
from lib.utils.worker_utils import decode_video, run_command


class MakeDecodable(Worker, CtxInterface):
    make_decodable_path: MakeDecodablePaths
    quality_list: list[str] = None
    decode_check = False
    cmd: str

    def init(self):
        self.make_decodable_path = MakeDecodablePaths(self.ctx)
        self.quality_list = ['0'] + self.ctx.quality_list

    def main(self):
        self.init()
        for _ in self.iterate_name_projection_tiling_tile_quality_chunk():
            with task(self):
                self.work()

    def work(self):
        self.check_decodable()
        self.check_dash()
        self.make_decodable_cmd()
        self.run()

    def run(self):
        run_command(self.cmd, folder=self.decodable_folder,
                    ui_prefix='\t')

    def make_decodable_cmd(self):
        self.cmd = (f'bash -c "cat {self.dash_init.as_posix()} {self.dash_m4s.as_posix()} '
                    f'> {self.decodable_chunk.as_posix()}"')

    def check_dash(self):
        msg = []
        if not self.dash_m4s.exists():
            msg += ['dash_m4s not exist']
            self.logger.register_log('Dash M4S not found.', self.dash_m4s)
        if not self.dash_init.exists():
            msg += ['dash_init not exist']
            self.logger.register_log('Dash init not found.', self.dash_m4s)
        if msg:
            raise AbortError('/'.join(msg))

    def check_decodable(self):
        try:
            chunk_size = self.decodable_chunk.stat().st_size
            if self.decode_check:
                self.check_one_chunk_decode()
        except FileNotFoundError:
            return 'decodable_chunk not found. Continue.'
        if chunk_size == 0:
            self.logger.register_log('Chunk size is 0.', self.decodable_chunk)
            self.decodable_chunk.unlink()
            return 'decodable_chunk size 0. Cleaning and continue.'
        raise AbortError(f'decodable_chunk is OK.')

    def check_one_chunk_decode(self):
        if self.status.get_status('decode_check_ok'):
            raise AbortError(f'Decoding check is ok.')

        print(f'\tDecoding check')
        decodable_chunk = self.decodable_chunk
        stdout = decode_video(decodable_chunk, ui_prefix='\t')
        if "frame=   30" not in stdout:  # specific for ffmpeg 5.0
            self.logger.register_log(f'Chunk Decode Error.', decodable_chunk)
            raise FileNotFoundError(f'Chunk Decode Error.')

    @property
    def decodable_folder(self):
        return self.make_decodable_path.decodable_folder

    @property
    def dash_init(self):
        return self.make_decodable_path.dash_init

    @property
    def dash_m4s(self):
        return self.make_decodable_path.dash_m4s

    @property
    def decodable_chunk(self):
        return self.make_decodable_path.decodable_chunk
