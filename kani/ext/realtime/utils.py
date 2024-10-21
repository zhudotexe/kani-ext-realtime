import asyncio
import base64
import json
import threading

from pydub import AudioSegment
from pydub.playback import play


# ===== audio =====
class AudioManager:
    def __init__(self):
        self.pending_segment: AudioSegment | None = None
        self.thread = None
        self._has_pending = threading.Event()
        self._lock = threading.Lock()

    def play(self, segment: AudioSegment):
        # start the thread to play in the bg
        if self.thread is None:
            self.thread = threading.Thread(target=self._thread_entrypoint, daemon=True)
            self.thread.start()
        # then do bookkeeping
        with self._lock:
            if self.pending_segment is not None:
                self.pending_segment += segment
            else:
                self.pending_segment = segment
                self._has_pending.set()

    def _thread_entrypoint(self):
        while True:
            self._has_pending.wait()
            with self._lock:
                segment = self.pending_segment
                self.pending_segment = None
                self._has_pending.clear()
            play(segment)


# def _play_with_ffplay(seg):
#     subprocess.call(["ffplay", "-nodisp", "-autoexit", "-hide_banner", "-"])
#
# 
# def _play_with_pyaudio(seg):
#     import pyaudio
#
#     p = pyaudio.PyAudio()
#     stream = p.open(format=p.get_format_from_width(seg.sample_width),
#                     channels=seg.channels,
#                     rate=seg.frame_rate,
#                     output=True)
#
#     # Just in case there were any exceptions/interrupts, we release the resource
#     # So as not to raise OSError: Device Unavailable should play() be used again
#     try:
#         # break audio into half-second chunks (to allows keyboard interrupts)
#         for chunk in make_chunks(seg, 500):
#             stream.write(chunk._data)
#     finally:
#         stream.stop_stream()
#         stream.close()
#
#         p.terminate()

_global_audio_manager = AudioManager()


async def play_audio(audio_bytes: bytes):
    """
    Play the given audio at the next available opportunity, using a global audio queue.

    This is a callback that should be passed to :meth:`.OpenAIRealtimeKani.full_round_stream` or
    :meth:`.OpenAIRealtimeKani.chat_round_stream` as the ``audio_callback`` parameter.
    """
    # we assume we're running in "raw 16 bit PCM audio at 24kHz, 1 channel, little-endian" mode
    # if we're in G.711 this will probably break
    audio = AudioSegment(data=audio_bytes, sample_width=2, channels=1, frame_rate=24000)
    asyncio.get_event_loop().run_in_executor(None, _global_audio_manager.play, audio)


def audio_to_item_create_event(audio_bytes: bytes) -> str:
    # adapted from OpenAI docs
    audio = AudioSegment(data=audio_bytes, sample_width=2, channels=1, frame_rate=24000)
    pcm_audio = audio.raw_data

    # # Load the audio file from the byte stream
    # audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # # Resample to 24kHz mono pcm16
    # pcm_audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data

    # Encode to base64 string
    pcm_base64 = base64.b64encode(pcm_audio).decode()

    event = {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": pcm_base64}]},
    }
    return json.dumps(event)
