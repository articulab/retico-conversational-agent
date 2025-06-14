import wave
import time

import retico_core
from retico_conversational_agent.Speaker_DM import SpeakerDmModule
from retico_conversational_agent.additional_IUs import TextAlignedAudioIU  

WAV_FILE = "tests/Hello1.wav"

def main():
    # Read WAV file
    with wave.open(WAV_FILE, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        audio_bytes = wf.readframes(wf.getnframes())
        nframes = wf.getnframes()

    # Create SpeakerDmModule
    frame_length = 0.2
    speaker = SpeakerDmModule(
        rate=rate,
        frame_length=frame_length,
        channels=channels,
        sample_width=sample_width,
        device_index=0, 
    )
    speaker.setup()
    speaker.prepare_run()

    # Split audio into frames and send as TextAlignedAudioIUs
    bytes_per_frame = int(rate * frame_length) * sample_width * channels
    total_frames = len(audio_bytes) // bytes_per_frame
    for i in range(0, len(audio_bytes), bytes_per_frame):
        chunk = audio_bytes[i:i+bytes_per_frame]
        iu = TextAlignedAudioIU(
            raw_audio=chunk,
            nframes=len(chunk) // (sample_width * channels),
            rate=rate,
            sample_width=sample_width,
            grounded_word="test",
            word_id=0,
            char_id=0,
            turn_id=0,
            clause_id=0,
            final=(i + bytes_per_frame >= len(audio_bytes)),  # True for last chunk
        )
        print("Sending IU, chunk size:", len(chunk))
        um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
        speaker.process_update(um)
        print("IU sent, buffer size now:", len(speaker.audio_iu_buffer))
        time.sleep(frame_length)

    print("Playing audio. Waiting for playback to finish...")
    time.sleep(1)
    speaker.shutdown()
    print("Done.")

if __name__ == "__main__":
    main()