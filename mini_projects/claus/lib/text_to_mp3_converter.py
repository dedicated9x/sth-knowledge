import boto3
import re
from mini_projects.claus.config import PROSODY_RATE, PATH_TO_AUTH
from mini_projects.claus.lib.clipboard_controller import ClipboardController


class TextToMp3Converter:
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

    @classmethod
    def convert(cls, text_, voice_id='Hans', is_simple=True):
        auth = cls._read_auth(PATH_TO_AUTH)
        polly_client = boto3.Session(**auth, region_name='us-west-2').client('polly')
        if is_simple:
            text = f'<speak><prosody rate="{PROSODY_RATE}%">{text_}</prosody></speak>'
        else:
            text = text_
        response = polly_client.synthesize_speech(
            VoiceId=voice_id, OutputFormat='mp3', Text=text, TextType='ssml'
        )
        sound = response['AudioStream'].read()
        return sound
