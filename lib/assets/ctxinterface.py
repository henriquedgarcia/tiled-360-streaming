from lib.assets.context import Context


class CtxInterface:
    ctx: Context

    @property
    def name_list(self):
        return self.ctx.name_list

    @property
    def projection_list(self):
        return self.ctx.projection_list

    @property
    def quality_list(self):
        return self.ctx.quality_list

    @property
    def tiling_list(self):
        return self.ctx.tiling_list

    @property
    def tile_list(self):
        return self.ctx.tile_list

    @property
    def chunk_list(self):
        return self.ctx.chunk_list

    @property
    def metric_list(self):
        return ["ssim", "mse", "s-mse", "ws-mse"]

    @property
    def users_list(self):
        return self.ctx.users_list

    @property
    def group_list(self):
        return self.ctx.group_list

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
    def n_tiles(self):
        return self.ctx.n_tiles

    @property
    def video_list_by_group(self):
        """

        :return: a dict like {group: video_list}
        """
        b = {group: [name for name in self.name_list
                     if self.ctx.config.videos_dict[name]['group'] == group]
             for group in self.group_list}
        return b
