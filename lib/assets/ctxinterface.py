from functools import cached_property

from lib.assets.context import Context
from lib.utils.util import make_tile_position_dict


class Factors:
    ctx: Context

    @property
    def name(self):
        return self.ctx.name

    @name.setter
    def name(self, value):
        self.ctx.name = value

    @property
    def projection(self):
        return self.ctx.projection

    @projection.setter
    def projection(self, value):
        self.ctx.projection = value

    @property
    def quality(self):
        return self.ctx.quality

    @quality.setter
    def quality(self, value):
        self.ctx.quality = value

    @property
    def tiling(self) -> str:
        return self.ctx.tiling

    @tiling.setter
    def tiling(self, value: str):
        self.ctx.tiling = value

    @property
    def tile(self) -> str:
        return self.ctx.tile

    @tile.setter
    def tile(self, value: str):
        self.ctx.tile = value

    @property
    def chunk(self):
        return self.ctx.chunk

    @chunk.setter
    def chunk(self, value):
        self.ctx.chunk = value

    @property
    def metric(self):
        return self.ctx.metric

    @metric.setter
    def metric(self, value):
        self.ctx.metric = value

    @property
    def user(self):
        return self.ctx.user

    @user.setter
    def user(self, value):
        self.ctx.user = value

    @property
    def group(self):
        return self.ctx.group

    @group.setter
    def group(self, value):
        self.ctx.group = value

    @property
    def turn(self):
        return self.ctx.turn

    @turn.setter
    def turn(self, value):
        self.ctx.turn = value

    @property
    def frame(self) -> int:
        return self.ctx.frame

    @frame.setter
    def frame(self, value: int):
        self.ctx.frame = value


class Lists:
    ctx: Context
    _name_list = None

    @cached_property
    def name_list(self) -> list[str]:
        return self.ctx.name_list

    @cached_property
    def projection_list(self):
        return self.ctx.projection_list

    _quality_list = None

    @property
    def quality_list(self):
        if self._quality_list is None:
            return self.ctx.quality_list
        return self._quality_list

    @quality_list.setter
    def quality_list(self, value):
        self._quality_list = value

    _tiling_list = None

    @property
    def tiling_list(self):
        if self._tiling_list is None:
            return self.ctx.tiling_list
        return self._tiling_list

    @tiling_list.setter
    def tiling_list(self, value):
        self._tiling_list = value

    @property
    def tile_list(self):
        return self.ctx.tile_list

    @cached_property
    def chunk_list(self):
        return self.ctx.chunk_list

    @cached_property
    def metric_list(self):
        return ['time', 'rate', "ssim", "mse", "s-mse", "ws-mse"]

    @property
    def users_list_by_name(self):
        return self.ctx.users_list_by_name

    @property
    def name_list_by_user(self):
        return self.ctx.name_list_by_user

    @cached_property
    def group_list(self):
        return self.ctx.group_list


class CtxInterface(Factors, Lists):
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
    def fov_res(self):
        return self.ctx.config.fov_res

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

    @cached_property
    def tile_position_dict(self) -> dict:
        """
        tile_position_dict[resolution: str][tiling: str][tile: str]
        :return:
        """
        return make_tile_position_dict(self.video_shape, self.tiling_list)
