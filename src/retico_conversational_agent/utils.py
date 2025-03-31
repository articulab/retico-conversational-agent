import numpy as np
import pydub
import torch


# DEVICE DEF
def device_definition(device):
    """Checks if the desired `device` is available, and if not, returns a default device (cpu).

    Args:
        device (str): the name of the desired device to use.

    Returns:
        str: the device that will be used.
    """
    cuda_available = torch.cuda.is_available()
    final_device = None
    if device is None:
        if cuda_available:
            final_device = "cuda"
        else:
            final_device = "cpu"
    elif device == "cuda":
        if cuda_available:
            final_device = "cuda"
        else:
            print(
                "device defined for instantiation is cuda but cuda is not available. Check your\
                cuda installation if you want the module to run on GPU. The module will run on\
                CPU instead."
            )
            # Raise Exception("device defined for instantiation is cuda but cuda is not available.
            # check you cuda installation or change device to "cpu")
            final_device = "cpu"
    elif device == "cpu":
        if cuda_available:
            print(
                "cuda is available, you can run the module on GPU by changing the device parameter\
                to cuda."
            )
        final_device = "cpu"
    return final_device


import audioop
import glob
import os
import wave
import librosa
import soundfile as sf


def resample_audio_file(src: str, dst: str, outrate: int = 16000):
    """Resample the audio's frame_rate to correspond to
    self.target_framerate.

    Args:
        src (str): source file to resample
        dst (_type_): destination file to write resampled audio in
        outrate (int, optional): The target samplerate. Defaults to 16000.
    """
    if not os.path.exists(src):
        print("Source not found!")
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        audio, sr = librosa.load(src, sr=None)
        print(sr)
    except:
        print("Failed to open source file!")
        return False

    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=outrate)

    try:
        sf.write(dst, resampled_audio, outrate)
    except:
        print("Failed to write wav")
        return False


def resample_audio(raw_audio: bytes, inrate: int, outrate: int):
    """Resample the audio's frame_rate to correspond to outrate.

    Args:
        raw_audio (bytes): the audio received from the microphone that
            could need resampling.
        inrate (int): the original samplerate
        outrate (int): the target samplerate

    Returns:
        bytes: resampled audio bytes
    """
    audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    return librosa.resample(audio_np, orig_sr=inrate, target_sr=outrate)


def resample_audio_2(raw_audio: bytes, inrate: int, outrate: int, sample_width: int = 2, channels: int = 2):
    """Resample the audio's frame_rate to correspond to
    self.target_framerate.

    Args:
        raw_audio (bytes): the audio received from the microphone that
            could need resampling.
        inrate (int): the original samplerate
        outrate (int): the target samplerate
        sample_width (int, optional): orginal audio samplewidth. Defaults to 2.
        channels (int, optional): orginal audio nb channels. Defaults to 2.

    Returns:
        bytes: resampled audio bytes
    """
    if inrate != outrate:
        s = pydub.AudioSegment(
            raw_audio,
            sample_width=sample_width,
            channels=channels,
            frame_rate=inrate,
        )
        s = s.set_frame_rate(outrate)
        return s._data
    return raw_audio
