import simpleaudio as sa

class WavPlayer:
    @staticmethod
    def play(src):
        wave_obj = sa.WaveObject.from_wave_file(src)
        play_obj = wave_obj.play()
        play_obj.wait_done()
