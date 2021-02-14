import os
import boto3
import re
import pathlib as pl
from lib.clipboard_controller import ClipboardController
from lib.klaus_state_analyzer import KlausStateAnalyzer

class VoiceSynthetizer:

    # @staticmethod
    # def get_path_to_output():
    #     max_record_number = KlausStateAnalyzer().get_max_record_number()
    #     new_record_filename = f"{max_record_number + 1}.mp3"
    #     new_record_path = pl.Path(os.path.realpath(__file__)).parent.parent.joinpath("temp").joinpath(new_record_filename)
    #     return new_record_path

    @staticmethod
    def read_auth(path_to_auth):
        with open(path_to_auth, "r") as infile:
            lines = infile.read()
        tags = ["AWSAccessKeyId", "AWSSecretKey"]
        patterns = [re.compile(tag + r"=([\S]+)") for tag in tags]
        values = [pattern.search(lines).group(1) for pattern in patterns]
        keys = ["aws_access_key_id", "aws_secret_access_key"]
        auth = dict(zip(keys, values))
        return auth

    def make_sound_from_text(self, path_to_auth, prosody_rate):
        # input_ = self.get_clipboard_value()
        input_ = ClipboardController.get_clipboard_value()
        auth = self.read_auth(path_to_auth)
        polly_client = boto3.Session(**auth, region_name='us-west-2').client('polly')
        text = f'<speak><prosody rate="{prosody_rate}%">{input_}</prosody></speak>'
        response = polly_client.synthesize_speech(
            VoiceId='Hans', OutputFormat='mp3', Text=text, TextType='ssml'
        )
        sound = response['AudioStream'].read()

        # new_record_path = self.get_path_to_output()
        # return new_record_path, sound
        return sound


# TODO get path to output powinno byc robione pozniej

# TODO ten clipboard to powinna byc inna libka