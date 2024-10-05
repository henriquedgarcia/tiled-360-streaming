from pathlib import Path

from lib.assets.ctxinterface import CtxInterface


class ViewportPSNRPaths(CtxInterface):
    @property
    def viewport_psnr_path(self) -> Path:
        folder = self.projectionect_path / 'ViewportPSNR'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def viewport_psnr_file(self) -> Path:
        folder = self.viewport_psnr_path / f'{self.projection}_{self.name}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"user{self.user}_{self.tiling}.json"
