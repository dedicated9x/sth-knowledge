# <amazon:breath duration="medium" volume="x-loud"/>
# <break time="1s"/>
# <prosody rate="90%"> </prosody>

def input2ssml(input_):
    with_breaks = [line + f'<break time="{sec}s"/>' for line, sec in input_]
    ssml = f'<speak><amazon:effect name="whispered">{"".join(with_breaks)}</amazon:effect></speak>'
    return ssml