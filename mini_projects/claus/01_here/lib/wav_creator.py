import os
import shutil
from lib.klaus_state_analyzer import KlausStateAnalyzer
from lib.paths_registry import PathsRegistry


class WavCreator:
    @staticmethod
    def _get_path_to_output():
        max_record_number = KlausStateAnalyzer().get_max_record_number()
        new_record_filename = f"{max_record_number + 1}.mp3"
        new_record_path = PathsRegistry.temp.joinpath(new_record_filename)
        return new_record_path

    @staticmethod
    def _tidy():
        path_to_temp = PathsRegistry.temp
        shutil.rmtree(path_to_temp)
        path_to_temp.mkdir()

    @staticmethod
    def _save_mp3(output_path, sound):
        file = open(output_path, 'wb')
        file.write(sound)
        file.close()

    @staticmethod
    def _convert_mp3_to_wav(src, dst):
        path_to_fffmpeg = PathsRegistry.ffmpeg
        os.system(f"{path_to_fffmpeg} -i {src} -acodec pcm_s16le -ac 1 -ar 16000 {dst}")

    @staticmethod
    def create_wav(sound):
        path_to_mp3 = WavCreator._get_path_to_output()
        path_to_wav = path_to_mp3.with_suffix(".wav")

        WavCreator._tidy()
        WavCreator._save_mp3(path_to_mp3, sound)
        WavCreator._convert_mp3_to_wav(path_to_mp3, path_to_wav)
        return path_to_wav