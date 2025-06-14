import wave
import threading
import time

from retico_core.audio import MicrophoneModule

RATE = 48000
FRAME_LENGTH = 0.02
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (16-bit PCM)
OUTPUT_FILE = "tests/mic_test.wav"

audio_frames = []

def audio_collector(mic):
    while mic._is_running:
        try:
            # process_update returns an UpdateMessage or None
            um = mic.process_update(None)
            if um is not None:
                for iu in um.incremental_units():
                    if hasattr(iu, "raw_audio"):
                        audio_frames.append(iu.raw_audio)
        except Exception:
            pass
        time.sleep(FRAME_LENGTH / 2)

def main():
    mic = MicrophoneModule(rate=RATE, frame_length=FRAME_LENGTH)
    mic.setup()
    mic.prepare_run()
    mic._is_running = True

    collector_thread = threading.Thread(target=audio_collector, args=(mic,))
    collector_thread.start()

    print("Recording from microphone. Press Enter to stop.")
    input()

    mic._is_running = False
    if mic.stream is not None:
        mic.shutdown()
    collector_thread.join()

    # Concatenate all audio frames
    audio_bytes = b"".join(audio_frames)

    # Save as WAV
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)

    print(f"Saved recording to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()