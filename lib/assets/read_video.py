import subprocess as sp

import numpy as np

from lib.utils.util import get_video_resolution, splitx


class ReadVideo:
    # Coeficientes de luminância perceptual (sRGB - ITU-R BT.709 )
    # BT.709 : Parameter values for the HDTV standards for production and international programme exchange
    # https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en

    def __init__(self, file_path, gray=True):
        self.file_path = file_path
        self.gray = gray

        self.largura, self.altura = get_video_resolution(file_path)
        self.frame_size = self.largura * self.altura * 3  # RGB24 = 3 bytes por pixel
        cmd = ''
        self.pipe = sp.Popen(cmd, stdout=sp.PIPE, bufsize=-1)

    def read_video(self):
        """
        Read frames from a video stream and yield them sequentially.

        This method reads raw video frames from the stdout of a subprocess pipe,
        processes the frames into either RGB or grayscale format (depending on the
        gray attribute), and yields ready-to-use frames. It stops when the pipe
        does not have sufficient data to construct a full frame, which could signal
        either the end of the video or an unexpected error.

        :yield: Processed video frames as NumPy arrays.
        :rtype: numpy.ndarray
        """
        while True:
            raw_frame = self.pipe.stdout.read(self.frame_size)
            self.pipe.stdout.close()
            self.pipe.wait()
            if len(raw_frame) < self.frame_size:
                break  # Fim do vídeo ou erro

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self.altura, self.largura, 3))

            if self.gray:
                frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
                frame = frame.astype(np.uint8)
            yield frame

        self.pipe.stdout.close()
        self.pipe.wait()

    def get_video_resolution(self):
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0:s=x',
            f'{self.file_path}'
        ]
        resolution = sp.check_output(cmd).decode().strip()
        largura, altura = splitx(resolution)
        return largura, altura
