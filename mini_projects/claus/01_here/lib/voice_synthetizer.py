import boto3
import re
from lib.clipboard_controller import ClipboardController


class VoiceSynthetizer:
    @staticmethod
    def _read_auth(path_to_auth):
        with open(path_to_auth, "r") as infile:
            lines = infile.read()
        tags = ["AWSAccessKeyId", "AWSSecretKey"]
        patterns = [re.compile(tag + r"=([\S]+)") for tag in tags]
        values = [pattern.search(lines).group(1) for pattern in patterns]
        keys = ["aws_access_key_id", "aws_secret_access_key"]
        auth = dict(zip(keys, values))
        return auth

    def make_sound_from_text(self, path_to_auth, prosody_rate):
        input_ = ClipboardController.get_clipboard_value()
        auth = self._read_auth(path_to_auth)
        polly_client = boto3.Session(**auth, region_name='us-west-2').client('polly')
        text = f'<speak><prosody rate="{prosody_rate}%">{input_}</prosody></speak>'
        response = polly_client.synthesize_speech(
            VoiceId='Hans', OutputFormat='mp3', Text=text, TextType='ssml'
        )
        sound = response['AudioStream'].read()
        return sound
