import asyncio
import base64
import json
import queue
import subprocess
import threading
import time
import warnings

from pydub import AudioSegment


# ===== audio =====
class AudioManagerBase:
    def play(self, segment: AudioSegment):
        raise NotImplementedError


class PyAudioAudioManager(AudioManagerBase):
    """Audio manager using a PyAudio stream"""

    def __init__(self):
        self.q = queue.Queue()
        self.thread = None
        self.stream = None

    def play(self, segment: AudioSegment):
        # push the segment onto the queue
        self.q.put(segment)
        # open the stream
        if self.stream is None:
            p = pyaudio.PyAudio()
            self.stream = p.open(
                format=p.get_format_from_width(segment.sample_width),
                channels=segment.channels,
                rate=segment.frame_rate,
                output=True,
            )
        # start the thread to handle the queue
        if self.thread is None:
            self.thread = threading.Thread(target=self._thread_entrypoint, daemon=True)
            self.thread.start()

    def _thread_entrypoint(self):
        from pydub.utils import make_chunks

        # hack: 100ms sleep before reading from q to avoid startup crunchiness
        time.sleep(0.1)
        while True:
            segment = self.q.get()
            for chunk in make_chunks(segment, 500):
                self.stream.write(chunk.raw_data)


# class FFMPEGAudioManager(AudioManagerBase):
#     """Audio manager using a ffplay process with a byte pipe"""
#
#     def __init__(self):
#         self._lock = threading.Lock()
#         self.ffmpeg = None
#         self.ffplay = None
#
#     def play(self, segment: AudioSegment):
#         # start the ffplay process to consume from our byte pipe
#         # TODO fixme - seems like ffplay continues to seek if no data from pipe, won't play new data until it
#         # "catches up" with the seek. Maybe piping to ffmpeg and having ffmpeg merge with silence can help?
#         with self._lock:
#             if self.ffmpeg is None:
#                 self.ffmpeg = subprocess.Popen(
#                     [
#                         "ffmpeg",
#                         "-use_wallclock_as_timestamps",
#                         "true",
#                         "-f",
#                         "s16le",
#                         "-ar",
#                         "24000",
#                         "-ac",
#                         "1",
#                         # "-re",
#                         "-i",
#                         "pipe:0",
#                         # "-af",
#                         # "aresample=async=1",
#                         "-f",
#                         "wav",
#                         "pipe:",
#                     ],
#                     stdin=subprocess.PIPE,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.DEVNULL,
#                 )
#                 self.ffplay = subprocess.Popen(
#                     ["ffplay", "-nodisp", "-i", "-"],
#                     stdin=self.ffmpeg.stdout,
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                 )
#                 # self.ffplay = subprocess.Popen(
#                 #     ["ffplay", "-nodisp", "-f", "s16le", "-ar", "24000", "-acodec", "pcm_s16le", "-i", "-"],
#                 #     stdin=self.ffmpeg.stdout,
#                 #     stdout=subprocess.DEVNULL,
#                 #     stderr=subprocess.DEVNULL,
#                 # )
#         # then send the bytes over the pipe
#         self.ffmpeg.stdin.write(segment.raw_data)
#         self.ffmpeg.stdin.flush()


class PyDubAudioManager(AudioManagerBase):
    """Fallback audio manager using pydub's default play if we don't have ffplay or pyaudio"""

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
        from pydub.playback import play

        while True:
            self._has_pending.wait()
            with self._lock:
                segment = self.pending_segment
                self.pending_segment = None
                self._has_pending.clear()
            play(segment)


try:
    import pyaudio

    _global_audio_manager = PyAudioAudioManager()
except ImportError:
    # # check if ffplay is available
    # _ffplay_available = (
    #     subprocess.run(["ffplay", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    # )
    # if _ffplay_available:
    #     _global_audio_manager = FFMPEGAudioManager()
    # else:
    #     warnings.warn(
    #         "You do not have PyAudio or ffmpeg installed. Playback from utilities like chat_in_terminal_audio may have"
    #         " choppy output. We recommend installing PyAudio or ffmpeg for best performance, but it is unnecessary if"
    #         " you are not playing audio on this machine."
    #     )
    warnings.warn(
        "You do not have PyAudio installed. Playback from utilities like chat_in_terminal_audio may have choppy"
        " output. We recommend installing PyAudio for best audio performance, but it is unnecessary if you are not"
        " playing audio on this machine."
    )
    _global_audio_manager = PyDubAudioManager()


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
