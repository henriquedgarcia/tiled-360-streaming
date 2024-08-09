from typing import Any

from lib.assets.worker import Worker


class RenamerAndCheck(Worker):
    def main(self):
        for self.video in self.video_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.turn in range(self.decoding_num):
                        for self.tile in self.tile_list:
                            self.compressed()

                            for self.chunk in self.chunk_list:
                                self.segmented()
                                pass

    def compressed(self) -> Any:
        folder = self.project_path / self.compressed_folder / self.basename
        old_name = folder / f'tile{self.tile}.mp4'
        if folder.exists():
            if old_name.exists() and not self.compressed_file.exists():
                old_name.replace(self.compressed_file)
                old_name.with_suffix('.log').replace(self.compressed_log)

            try:
                folder.rmdir()
            except OSError:
                pass

    def segmented(self):
        folder = self.project_path / self.segment_folder / self.basename
        if folder.exists():
            old_name = folder / f'tile{self.tile}.log'

            if old_name.exists() and not self.segment_log.exists():
                old_name.replace(self.segment_log)

            for self.chunk in self.chunk_list:
                chunk = int(str(self.chunk))
                old_file = folder / f'tile{self.tile}_{chunk:03d}.mp4'
                if old_file.exists() and not self.segment_file.exists():
                    old_file.replace(self.segment_file)

            try:
                folder.rmdir()
            except OSError:
                pass
