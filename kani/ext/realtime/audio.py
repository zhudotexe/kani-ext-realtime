import asyncio
import base64
import io
import queue
import subprocess
import threading
import time
import warnings
from typing import AsyncIterable

from pydub import AudioSegment


# ===== output =====
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


class FFMPEGAudioManager(AudioManagerBase):
    """Audio manager using a ffplay process with a byte pipe"""

    def __init__(self):
        self.q = queue.Queue()
        self.ffplay = None
        self.thread = None

    def play(self, segment: AudioSegment):
        # push the segment onto the queue
        self.q.put(segment)
        # open the stream
        if self.ffplay is None:
            self.ffplay = subprocess.Popen(
                ["ffplay", "-nodisp", "-f", "s16le", "-ar", "24000", "-acodec", "pcm_s16le", "-i", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # start the thread to handle the queue
        if self.thread is None:
            self.thread = threading.Thread(target=self._thread_entrypoint, daemon=True)
            self.thread.start()

    def _thread_entrypoint(self):
        # 50ms of silence
        # 16b frame * 24k fps = 2 * 24000 / 20 bytes = 2400B
        silence_bytes = b"\0\0" * 2400
        playing_until = time.perf_counter()
        while True:
            try:
                # if we have a segment, write it and wait for its duration before writing more
                segment = self.q.get(block=False)
                self.ffplay.stdin.write(segment.raw_data)
                self.ffplay.stdin.flush()
                playing_until += segment.duration_seconds
            except queue.Empty:
                now = time.perf_counter()
                # if we are currently playing audio, wait a bit and check if we have more to do once it's half done
                if playing_until > now:
                    time.sleep(max(0.05, (playing_until - now) / 2))
                # otherwise write silence
                else:
                    # no lag time needed - processing should happen within 41us so the next frame is ready in time
                    self.ffplay.stdin.write(silence_bytes)
                    self.ffplay.stdin.flush()
                    time.sleep(0.05)
                    playing_until = time.perf_counter()


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
    _has_pyaudio = True
except ImportError:
    _has_pyaudio = False
    # check if ffplay is available
    _ffplay_available = (
        subprocess.run(["ffplay", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    )
    if _ffplay_available:
        _global_audio_manager = FFMPEGAudioManager()
    else:
        warnings.warn(
            "You do not have PyAudio or ffmpeg installed. Playback from utilities like chat_in_terminal_audio may have"
            " choppy output. We recommend installing PyAudio or ffmpeg for best playback performance, but it is"
            " unnecessary if you are not playing audio on this machine."
        )
        _global_audio_manager = PyDubAudioManager()


async def play_audio(audio_bytes: bytes):
    """
    Play the given audio at the next available opportunity, using a global audio queue.

    This is a callback that should be passed to :meth:`~kani.Kani.full_round_stream` or
    :meth:`~kani.Kani.chat_round_stream`, or :meth:`.OpenAIRealtimeKani.full_duplex` as the ``audio_callback``
    parameter.
    """
    # we assume we're running in "raw 16 bit PCM audio at 24kHz, 1 channel, little-endian" mode
    # if we're in G.711 this will probably break
    audio = AudioSegment(data=audio_bytes, sample_width=2, channels=1, frame_rate=24000)
    asyncio.get_event_loop().run_in_executor(None, _global_audio_manager.play, audio)


# ===== input =====
def audio_to_b64(audio_bytes: bytes) -> str:
    """Encode an arbitrarily-encoded audio bytestring into the correct format."""
    # Load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # Resample to 24kHz mono pcm16
    pcm_audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data

    # Encode to base64 string
    pcm_base64 = base64.b64encode(pcm_audio).decode()
    return pcm_base64


if _has_pyaudio:

    class PyAudioInputManager:
        """Audio manager using a PyAudio stream. This class should NOT be constructed manually."""

        def __init__(self, mic_id: int | None):
            self.q = asyncio.Queue()
            self.loop = asyncio.get_event_loop()

            # init pyaudio, create a recording stream
            p = pyaudio.PyAudio()
            self.stream = p.open(
                format=p.get_format_from_width(2),
                channels=1,
                rate=24000,
                frames_per_buffer=1200,
                input=True,
                input_device_index=mic_id,
            )

            # launch thread to start streaming from it
            self.thread = threading.Thread(target=self._thread_entrypoint, daemon=True)
            self.thread.start()

        def _thread_entrypoint(self):
            while True:
                n_available = self.stream.get_read_available()
                if not n_available:
                    time.sleep(0.05)
                    continue
                frame = self.stream.read(n_available, exception_on_overflow=False)
                fut = asyncio.run_coroutine_threadsafe(self.q.put(frame), self.loop)
                fut.result()

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.q.get()

    def get_audio_stream(mic_id: int | None) -> AsyncIterable[bytes]:
        """Return an audio stream manager that yields audio frames from the given mic."""
        return PyAudioInputManager(mic_id)

    def list_mics():
        """Print a list of all microphones connected to this device."""
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        n_devices = info.get("deviceCount")
        for i in range(0, n_devices):
            if (p.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels")) > 0:
                print(f"ID: {i} -- {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

else:

    def _missing(*_, **__):
        raise ImportError(
            "You must install PyAudio to record from the mic. You can install this"
            ' with `pip install "kani-ext-realtime[all]"`.'
        )

    get_audio_stream = _missing
    list_mics = _missing
