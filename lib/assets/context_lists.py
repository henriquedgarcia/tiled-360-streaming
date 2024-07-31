from factors import Factors


class ContextLists(Factors):
    @property
    def video_list(self):
        for video in self.videos_dict:
            yield video

    @property
    def tiling_list(self):
        for tiling in self._tiling_list:
            yield tiling

    @property
    def tile_list(self) -> str:
        for tile in range(self.n_tiles):
            yield str(tile)

    @property
    def quality_list(self):
        for quality in self._quality_list:
            yield quality

    @property
    def chunk_list(self):
        for chunk in range(1, self.duration + 1):
            return str(chunk)

    @property
    def group_list(self):
        conjunto = set()
        for video in self.video_list:
            conjunto.update(self.videos_dict[video]['group'])
        return list(conjunto)

    @property
    def metric_list(self):
        return self.config_dict['metric_list']

    @property
    def user_list(self):
        for user in []:
            yield user

    @property
    def frame_list(self) -> list[str]:
        for frame in range(self.n_frames):
            yield str(frame)