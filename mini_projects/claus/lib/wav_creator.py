import os
import shutil
from mini_projects.claus.lib.paths_registry import PathsRegistry


class WavCreator:
    path_to_ffmpeg = PathsRegistry.ffmpeg

    def __init__(self, workdir):
        self.workdir = workdir

    def _tidy(self):
        shutil.rmtree(self.workdir)
        self.workdir.mkdir()

    @staticmethod
    def _save_mp3(output_path, sound):
        file = open(output_path, 'wb')
        file.write(sound)
        file.close()

    @classmethod
    def _convert_mp3_to_wav(cls, src, dst):
        os.system(f"{cls.path_to_ffmpeg} -i {src} -acodec pcm_s16le -ac 1 -ar 16000 {dst}")

    def create_wav(self, sound, filename):
        path_to_mp3 = self.workdir.joinpath(filename)
        path_to_wav = path_to_mp3.with_suffix(".wav")

        self._tidy()
        self._save_mp3(path_to_mp3, sound)
        self._convert_mp3_to_wav(path_to_mp3, path_to_wav)
        return path_to_wav