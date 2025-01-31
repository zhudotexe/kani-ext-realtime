import base64
import io

from pydub import AudioSegment


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
