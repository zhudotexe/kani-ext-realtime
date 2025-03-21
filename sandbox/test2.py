import base64
import faulthandler

from pydub import AudioSegment
from pydub.playback import play

faulthandler.enable()

if __name__ == "__main__":
    audio = "BQAEAP7/BgAAAAIABgAAAAgAAQAHAAEA+/8GAPv/CQD9/wQA///+/wQA9/8CAPz/AwABAP7/AwD7/wIA/f8AAP7//f/9//r/AAD7/wcA/P8AAAEA9/8EAPX/AgDv/wUA+v/8//7/9P/+//T////3//7/AAD9//v/+f/4//3/+v/7//z/+P/5//j/+v/8//7/+v/+//j//f/0//7/+//7//r/+P/7//X/+f/7//z////5//3/+f/2/wAA9P8BAPT/+v/1//H/AwDz////8f/+//H//f////b//v/u/wAA9f////L/9//2//D//P/w//v/7f/5//b/+P/0//P/8v/x//D/8P/1/+//9v/t//D/9f/1/+v/8//t//D/8v/4/+z/7v/r/+v/7f/s//D/6//x/+3/7v/v/+n/8f/u//P/7f/v//H/7P/y/+7/+//y//P/7//s//f/7//7/+z/+v/t//P/9f/y//v/8v/6//D//f/6//T//P/x//z/9v/8//n/+/8AAPz/9v/7//z/+v/+//v//P/1//r/+//+//7/+P8AAPv/AAD3////AQD5/wIA9v8FAPn/DQADAAUABQD5/wkA+v8SAPf/FwD//woABgACAAkA//8aAA0ADQAKAAsAEQABABcABQAZAAsAEAAPAAwAHwAIAB0ABAAcAA0AFwAXABMAHwAPACEAEwAfABQAFwAZABAAIAATAB0AEgAcABwAGAAjABYAHwARABgAFAAWABcAFgAcABEAHgARABoAFQAUAB0ADwAdAA4AGAAMABIAFQAHACEABAAiAA0AFwASABAAHQAEACAABQAcAAUAGQAVABIAFQALABUADQAXABMAEAAPABIAEAAUABAADAAKAA4ADQAOABIACwAKAA0ADQAOAA4ABwAJAAsACQAGAAoADwAKAAkABgAIAAQACAAGAAQABAABAAQA+f/9/wEA+//+//n////3//z/9f/1//z/9f8AAPD/+v/u//r/8f/3//H/8P/y/+3/9P/m//b/5f/2/+b/9P/n//X/7f/o//D/5v/y/+P/7//o//D/6f/n/+//6P/y/+v/3P/l/+b/7//t/+z/6//b/+7/2P/0/9//7v/a/+7/4P/e/+L/4//q/97/4//W/+b/2v/s/9L/5v/T/9j/1//X/9v/0v/h/9j/4P/W/9X/1v/Q/9X/0v/b/9D/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"
    audio_bytes = base64.b64decode(audio)
    # audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="u16le")
    # audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    audio = AudioSegment(data=audio_bytes, sample_width=2, channels=1, frame_rate=24000)
    play(audio)

"""
ffmpeg -i test.mp3 -f s16le -acodec pcm_s16le -ar 24000 -ac 1 - | ffplay -nodisp -f s16le -ar 24000 -ac 1 -

https://www.reddit.com/r/ffmpeg/comments/148o6jb/inserting_silence_into_stream_when_stream_is/
https://superuser.com/questions/1598926/fill-gaps-in-piped-in-audio-with-silence-in-ffmpeg
https://superuser.com/questions/1859542/ffplay-reading-pcm-data-from-pipe-pause-when-no-data-available-instead-of-cont
all in one? weird timing
ffmpeg -i test.mp3 -f s16le -acodec pcm_s16le -ar 24000 -ac 1 - | \
ffmpeg \
    -use_wallclock_as_timestamps true \
    -f s16le -ar 24000 -ac 1 -re -i pipe:0 -af aresample=async=1 \
    -f wav pipe: | \
ffplay -nodisp -

2 resamples? too fast
ffmpeg -i test.mp3 -f s16le -acodec pcm_s16le -ar 24000 -ac 1 - | \
ffmpeg \
    -f s16le -ar 24000 -ac 1 -i pipe:0 \
    -f wav pipe: | \
ffmpeg \
    -f wav -re -i pipe:0 \
    -f wav -af aresample=async=1 pipe: | \
ffplay -nodisp -

filters:
apad
amix
"""
