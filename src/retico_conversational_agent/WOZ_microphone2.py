"""
MicrophonePTTModule
==================

This module provides push-to-talk capabilities to the classic retico MicrophoneModule
which captures audio signal from the microphone and chunks the audio signal into AudioIUs.
"""

import queue
import time
import keyboard
import pyaudio
import wave
import scipy.io.wavfile as wav

import retico_core
from retico_core.audio import MicrophoneModule


class WOZMicrophoneModul2(MicrophoneModule):
    """A modules overrides the MicrophoneModule which captures audio signal from the microphone and chunks the audio signal into AudioIUs.
    The addition of this module is the introduction of the push-to-talk capacity : the microphone's audio signal is captured only while the M key is pressed.
    """

    @staticmethod
    def name():
        return "WozMicrophone Module"

    @staticmethod
    def description():
        return "A producing module that produce audio from wave file."

    def __init__(
        self,
        # file="audios/Recording.wav",
        file="audios/low.wav",
        frame_length=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file = file
        self.frame_length = frame_length
        self._run_thread_active = False
        self.args = None
        self.list_ius = []
        self.silence_ius = []
        self.cpt = 0
        self.play_audio = False

    def setup(self, **kwargs):
        super().setup(**kwargs)
        # load data
        frame_rate, data = wav.read(self.file)
        audio_data = data
        n_channels = 1 if len(data.shape) == 1 else data.shape[1]
        # sample_width = data.dtype.itemsize * 8
        sample_width = data.dtype.itemsize
        self.terminal_logger.info("sample_width", sample_width=sample_width, debug=True)
        sample_width = 2

        # wf = wave.open(self.file, "rb")
        # frame_rate = wf.getframerate()
        # n_channels = wf.getnchannels()
        # sample_width = wf.getsampwidth()
        # audio_data = wf.readframes(1000000)
        # # audio_data = wf.readframes(10000)
        # wf.close()

        # 12:53:05.57 [info     ] WozMicrophone load sound
        # chunk_size= | 320 |  debug= ( True )  frame_rate= | 16000 |  n_channels= | 1 |  rate= | 16000 |  sample_width= | 2 |  total_time= 1.82

        # calculate IUs
        rate = frame_rate * n_channels
        chunk_size = round(rate * self.frame_length)
        max_cpt = int(len(audio_data) / (chunk_size * sample_width))
        total_time = len(audio_data) / (rate * sample_width)
        self.terminal_logger.info(
            "load sound",
            debug=True,
            frame_rate=frame_rate,
            n_channels=n_channels,
            sample_width=sample_width,
            rate=rate,
            chunk_size=chunk_size,
            total_time=total_time,
        )
        self.list_ius = []
        read_cpt = 0
        while read_cpt < max_cpt:
            sample = audio_data[(chunk_size * sample_width) * read_cpt : (chunk_size * sample_width) * (read_cpt + 1)]
            read_cpt += 1
            output_iu = self.create_iu(
                audio=sample, raw_audio=sample, nframes=chunk_size, rate=rate, sample_width=sample_width
            )
            # output_iu.dispatch = True
            self.list_ius.append((output_iu, retico_core.UpdateType.ADD))

    def callback(self, in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.

        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
        """
        if keyboard.is_pressed("m"):
            self.play_audio = True
        if self.play_audio is True:
            in_data = self.list_ius[self.cpt][0].raw_audio
            if self.cpt == len(self.list_ius) - 1:
                self.cpt = 0
                self.play_audio = False
            else:
                self.cpt += 1
            # self.terminal_logger.info(len(in_data), debug=True)
            self.audio_buffer.put(in_data)
        else:
            self.audio_buffer.put(b"\x00" * self.sample_width * self.chunk_size)
        return (in_data, pyaudio.paContinue)

    def process_update(self, _):
        """overrides MicrophoneModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#202

        Returns:
            UpdateMessage: list of AudioIUs produced from the microphone's audio signal.
        """
        if not self.audio_buffer:
            return None
        try:
            sample = self.audio_buffer.get(timeout=1.0)
        except queue.Empty:
            return None
        # output_iu = self.create_iu()
        # output_iu.set_audio(sample, self.chunk_size, self.rate, self.sample_width)
        output_iu = self.create_iu(
            raw_audio=sample,
            nframes=self.chunk_size,
            rate=self.rate,
            sample_width=self.sample_width,
        )
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def shutdown(self):
        """Close the audio stream."""
        # super().shutdown()
        # self.stream.stop_stream()
        # self.stream.close()
        # self.stream = None
        # time.sleep(0.5)
        if self.stream:
            # if self.stream.is_active():
            #     self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.audio_buffer = queue.Queue()
