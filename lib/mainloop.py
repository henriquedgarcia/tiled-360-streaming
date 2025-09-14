import os
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path

from config.config import Config
from lib.assets.context import Context
from lib.assets.ctxinterface import ContextInterface
from lib.assets.errors import TilesOkError, DecodableOkError, DashNotOkError
from lib.assets.logger import Logger
from lib.assets.paths.dectimepaths import DectimePaths
from lib.assets.paths.make_decodable_paths import MakeDecodablePaths
from lib.assets.worker import Worker
from lib.utils.util import run_command, save_json

State = namedtuple(typename='State',
                   field_names=['name', 'projection', 'tiling',
                                'tile', 'quality', 'chunk',
                                'frame', 'rate_control'],
                   defaults=(None, None, None,
                             None, None, None,
                             None, None))

Task = namedtuple('Task',
                  ['func', 'kwargs'],
                  defaults=(None, None))


class DecodableError(Exception): ...


class TileLogCorruptError(Exception): ...


class CodecVersionError(Exception): ...


class ContextInterface(Factors, Lists, ABC):
    ctx: Context

    @property
    def attempt(self):
        return self.ctx.attempt

    @attempt.setter
    def attempt(self, value):
        self.ctx.attempt = value

    @property
    def video_shape(self):
        return self.ctx.video_shape

    @property
    def scale(self):
        return self.ctx.scale

    @property
    def vp_res(self):
        return self.ctx.config.vp_res

    @property
    def proj_res(self):
        return self.ctx.scale

    @property
    def fov(self):
        return self.ctx.fov

    @property
    def n_tiles(self):
        return self.ctx.n_tiles

    @property
    def n_frames(self):
        return self.config.n_frames

    @property
    def config(self):
        return self.ctx.config

    @property
    def fps(self):
        return self.config.fps

    @property
    def gop(self):
        return self.config.gop

    @property
    def rate_control(self):
        return self.config.rate_control

    @property
    def decoding_num(self):
        return self.config.decoding_num

    @property
    def dataset_name(self):
        return self.config.dataset_file

    @property
    def user_hmd_data(self) -> list:
        return self.ctx.hmd_dataset[self.name + '_nas'][self.user]

    @property
    def video_list_by_group(self):
        """

        :return: a dict like {group: video_list}
        """
        b = {group: [name for name in self.name_list
                     if self.config.videos_dict[name]['group'] == group]
             for group in self.group_list}
        return b

    @property
    def tile_position_dict(self) -> dict:
        """
        tile_position_dict[resolution: str][tiling: str][tile: str]
        :return:
        """
        from py360tools import ERP, CMP
        proj_obj = (ERP if self.projection == 'erp' else CMP)(proj_res=self.proj_res, tiling=self.tiling)

        return make_tile_position_dict(proj_obj, self.tiling_list)

class Work(ABC, ContextInterface, MakeDecodablePaths):
    logger: Logger

    def __init__(self, ctx, logger):
        self.ctx: Context = ctx
        self.logger: Logger = logger
        self.make_tiles = MakeTiles()

    @abstractmethod
    def work(self): ...

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.ctx.name_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project_folder: {self.config.project_folder}\n'
              f'  fps: {self.config.fps}\n'
              f'  gop: {self.config.gop}\n'
              f'  qualities: {self.ctx.quality_list}\n'
              f'  patterns: {self.ctx.tiling_list}')
        print('=' * 70)


class Make:
    name: str
    projection: str
    tiling: str
    tile: str
    quality: str
    chunk: str
    _state: State
    cmd: str

    def __init__(self, config: Config):
        self.project_folder = config.project_folder
        self.rate_control = config.rate_control
        self.msg_err = ''

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value: State):
        self.name = value.name
        self.projection = value.projection
        self.tiling = value.tiling
        self.tile = value.tile
        self.quality = value.quality
        self.rate_control = value.rate_control
        self.chunk = value.chunk
        self._state = value


class MakeDecodable(Make):
    def __call__(self, state: State):
        self.state = state
        if self.decodable_is_ok(): raise DecodableOkError(self.msg_err)
        if not self.dash_is_ok(): raise DashNotOkError(self.msg_err)
        self.make_decodable_cmd()
        self.run_command()
        # Colocar aqui um teste de decodificação
        return 0

    def decodable_is_ok(self):
        try:
            chunk_size = self.decodable_chunk.stat().st_size
            if chunk_size == 0:
                self.msg_err = f'chunk_size == 0: {self.decodable_chunk}'
                self.decodable_chunk.unlink()
                raise FileNotFoundError
        except FileNotFoundError:
            return False
        # colocar aqui um teste de decodificação
        return True

    def dash_is_ok(self):
        msg = []
        if not self.dash_m4s.exists():
            msg += ['dash_m4s not exist']
        if not self.dash_init.exists():
            msg += ['dash_init not exist']
        if msg:
            self.msg_err = '/'.join(msg)
            return False
        return True

    def make_decodable_cmd(self):
        self.cmd = (f'bash -c "cat {self.dash_init.as_posix()} {self.dash_m4s.as_posix()} '
                    f'> {self.decodable_chunk.as_posix()}"')

    def run_command(self):
        run_command(self.cmd, folder=self.decodable_folder, log_file=None,
                    ui_prefix='\t')

    @property
    def decodable_chunk(self):
        return (self.project_folder / 'decodable'
                / f'{self.name}'
                / f'{self.projection}'
                / f'{self.tiling}'
                / f'tile{self.tile}'
                / f'{self.rate_control}{self.quality}'
                / f'chunk{self.chunk}.mp4')

    @property
    def decodable_folder(self) -> Path:
        return self.decodable_chunk.parent

    @property
    def mpd_folder(self):
        return (self.project_folder / 'dash' /
                f'{self.name}/{self.projection}/{self.tiling}/{self.tile}')

    @property
    def dash_mpd(self):
        return self.mpd_folder / f'tile{self.tile}.mpd'

    @property
    def dash_init(self):
        return self.mpd_folder / f'tile{self.tile}_{self.rate_control}{self.quality}_init.mp4'

    @property
    def dash_m4s(self):
        return self.mpd_folder / f'tile{self.tile}_{self.rate_control}{self.quality}_{self.chunk}.m4s'


class ProduceDecodable(Work):
    make_decodable: MakeDecodable

    def work(self):
        self.make_decodable = MakeDecodable(self.ctx.config)
        for _ in self.main_iter():
            state = State(name=self.name,
                          projection=self.projection,
                          tiling=self.tiling,
                          tile=self.tile,
                          quality=self.quality,
                          chunk=self.chunk,
                          rate_control=self.rate_control)

            task = Task(self.make_decodable, state)
            self.fila.append(task, state)

    def main_iter(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        for self.quality in self.quality_list:
                            for self.chunk in self.chunk_list:
                                yield

    def do_tiles(self):
        for self.quality in self.quality_list:
            try:
                self.assert_tiles()
            except TilesOkError:
                scheduling(Dash)
                continue

            try:
                self.assert_lossless()
            except FileNotFoundError:
                self.logger.register_log('Tile log is corrupt', self.tile_log)

                'notting to do'
                continue

            cmd = self.make_tile_cmd()
            run_command(cmd=cmd, folder=self.tile_folder, log_file=self.tile_log,
                        ui_prefix='\t')
            yield

    def assert_tiles(self):
        try:
            self.check_tile_video()
            raise TilesOkError('')
        except FileNotFoundError:
            self.tile_log.unlink(missing_ok=True)
            self.tile_video.unlink(missing_ok=True)

    def check_tile_video(self):
        compressed_file_size = self.tile_video.stat().st_size
        compressed_log_text = self.tile_log.read_text()

        if compressed_file_size == 0:
            raise FileNotFoundError('Filesize is 0')

        if 'encoded 1800 frames' not in compressed_log_text:
            self.logger.register_log('Tile log is corrupt', self.tile_log)
            raise FileNotFoundError

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.logger.register_log('Codec version is different.', self.tile_log)
            raise FileNotFoundError

    def assert_lossless(self):
        try:
            lossless_video_size = self.lossless_video.stat().st_size
            if lossless_video_size == 0:
                raise FileNotFoundError('lossless_video not found.')

        except FileNotFoundError:
            self.logger.register_log('lossless_video not found.', self.lossless_video)
            raise AbortError(f'lossless_video not found.')

    except IndexError:
    return 'fila vazia'


class MakeTiles:
    name: str
    projection: str
    tiling: str
    tile: str
    quality: str

    def __init__(self):
        pass

    def work(self, state: State):
        self.name = state.name
        self.projection = state.projection
        self.tiling = state.tiling
        self.tile = state.tile
        self.quality = state.quality

        self.assert_tiles_not_exist()
        self.assert_lossless()

        cmd = self.make_tile_cmd()
        run_command(cmd=cmd, folder=self.tile_folder, log_file=self.tile_log,
                    ui_prefix='\t')

    def assert_tiles_not_exist(self):
        try:
            self.check_tile_video()
        except FileNotFoundError:
            self.tile_log.unlink(missing_ok=True)
            self.tile_video.unlink(missing_ok=True)
            return

    def check_tile_video(self):
        compressed_file_size = self.tile_video.stat().st_size
        compressed_log_text = self.tile_log.read_text()

        if compressed_file_size == 0:
            raise FileNotFoundError('Filesize is 0')

        if 'encoded 1800 frames' not in compressed_log_text:
            raise TileLogCorruptError

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            raise CodecVersionError
        raise TilesOkError('')

    def assert_lossless(self):
        try:
            lossless_video_size = self.lossless_video.stat().st_size
            if lossless_video_size == 0:
                raise FileNotFoundError('lossless_video not found.')

        except FileNotFoundError:
            self.logger.register_log('lossless_video not found.', self.lossless_video)
            raise AbortError(f'lossless_video not found.')

    def make_tile_cmd(self) -> str:
        x1, x2, y1, y2 = self.tile_position
        crop_params = f'crop=w={x2 - x1}:h={y2 - y1}:x={x1}:y={y1}'

        gop_options = f'keyint={self.gop}:min-keyint={self.gop}:open-gop=0'
        misc_options = f':scenecut=0:info=0'
        qp_options = ':ipratio=1:pbratio=1' if self.rate_control == 'qp' else ''
        lossless_option = ':lossless=1' if self.quality == '0' else ''
        codec_params = f'-x265-params {gop_options}{misc_options}{qp_options}{lossless_option}'
        codec = f'-c:v libx265'
        output_options = f'-{self.rate_control} {self.quality} -tune psnr'

        cmd = ('bash -c '
               '"'
               'bin/ffmpeg -hide_banner -y -psnr '
               f'-i {self.lossless_video.as_posix()} '
               f'{output_options} '
               f'{codec} {codec_params} '
               f'-vf {crop_params} '
               f'{self.tile_video.as_posix()}'
               f'"')

        return cmd

    @property
    def lossless_video(self) -> Path:
        return self.lossless_folder / self.projection / f'{self.name}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_video.with_suffix('.log')

    @property
    def tile_folder(self) -> Path:
        folder = self.tiles_folder / self.folder_name_proj_tiling_tile
        return folder

    @property
    def tile_video(self) -> Path:
        return self.tile_folder / f'tile{self.tile}_{self.rate_control}{self.ctx.quality}.mp4'

    @property
    def tile_log(self) -> Path:
        return self.tile_video.with_suffix('.log')


class Consumer(Work):
    def work(self):
        try:
            task, kwargs = self.fila.pop(0)
        except IndexError:
            return

        try:
            task(**kwargs)
        except DecodableError:
            self.fila.append((task, kwargs))

    def do_work(self, state):
        print(f'\rProcessing {state.name}/{state.projection}/{state.tiling}/{state.tile}', end='')


class MainLoop(Worker, DectimePaths):
    consumer: Work
    producer: Work

    def main_iter(self):
        for self.name in self.name_list:
            for self.projection in self.projection_list:
                for self.tiling in self.tiling_list:
                    for self.tile in self.tile_list:
                        yield State(name=self.name, projection=self.projection,
                                    tiling=self.tiling, tile=self.tile)

    def init(self):
        fila = []
        logger = Logger(self.ctx)
        self.consumer = Consumer(fila, logger)
        # self.producer = ProduceDecodable(fila, logger)

    def main(self):
        for state in self.main_iter():
            self.consumer.work(state)
            self.producer.work()

        main_state = State(self.name, self.projection, self.tiling, self.tile)
        self.fila.append(main_state)

    save_json()


if __name__ == '__main__':
    os.chdir('../')

    # config_file = Path('config/config_cmp_qp.json')
    # videos_file = Path('config/videos_reduced.json')

    config_file = Path('config/config_pres_qp.json')
    videos_file = Path('config/videos_pres.json')

    config = Config(config_file, videos_file)
    ctx = Context(config=config)

    MainLoop(ctx).run()
