import datetime
import threading
import time

import retico_core
from retico_core.audio import AudioIU, SpeechIU
from retico_conversational_agent.additional_IUs import TextAlignedAudioIU
from retico_core.log_utils import log_exception

import wave
import scipy.io.wavfile as wav

# import keyboard
from pynput.keyboard import Key, Listener


class WozMicrophoneModule(retico_core.AbstractProducingModule):
    # class WozMicrophoneModule(retico_core.AbstractModule):
    """A module that produces IUs containing audio signals that are captured from a wav file
    (it simulate the capture of audio by microphone with an already recorded audio saved in a wav file).
    """

    @staticmethod
    def name():
        return "WozMicrophone Module"

    @staticmethod
    def description():
        return "A producing module that produce audio from wave file."

    @staticmethod
    def output_iu():
        return AudioIU
        # return SpeechIU
        # return TextAlignedAudioIU

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

    def process_update(self, _):
        self.terminal_logger.info("PROCESS UPDATE")
        return None

    def setup(self):
        super().setup()
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

        # Add silence for VAD
        self.silence_ius = []
        silence_duration = 1
        nb_silence_IUs = round(silence_duration / self.frame_length)
        print("nb_silence_IUs = ", nb_silence_IUs)
        for i in range(nb_silence_IUs):
            silence_sample = b"\x00" * chunk_size * sample_width
            output_iu = self.create_iu(
                audio=silence_sample,
                raw_audio=silence_sample,
                nframes=rate * silence_duration,
                rate=rate,
                sample_width=sample_width,
            )
            # output_iu.dispatch = True
            self.silence_ius.append((output_iu, retico_core.UpdateType.ADD))
        # Register key listener
        # keyboard.on_press(self.on_key_event)
        # listener = Listener(on_press=self.on_press)
        # listener.start()

    def prepare_run(self):
        super().prepare_run()
        self._run_thread_active = True
        threading.Thread(target=self.key_listener, daemon=True).start()
        # threading.Thread(
        #     target=self.thread3,
        #     daemon=True,
        #     # args=args,
        # ).start()
        print("woz mic started")

    def send_audio(self):
        um = retico_core.UpdateMessage()
        um.add_ius(iu_list=self.list_ius)
        self.append(um)

        um = retico_core.UpdateMessage()
        um.add_ius(iu_list=self.silence_ius)
        self.append(um)

    def key_listener(self):
        listener = Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        if key == Key.esc:  # Stop on 'Esc' key
            print("ESC pressed, exiting...")
            return False
        elif key == Key.enter or (hasattr(key, "char") and key.char == "q"):
            print(f"Key {key} pressed! Sending audio...")
            self.send_audio()

    # def on_key_event(self, event):
    #     if event.name in ["q", "enter"]:
    #         print(f"Key {event.name} pressed! Sending audio...")
    #         self.send_audio()

    # def thread2(self):
    #     while self._run_thread_active:
    #         self.terminal_logger.info("Press Q to send audio", debug=True)
    #         # keyboard.wait("q")
    #         if keyboard.is_pressed("q") or keyboard.is_pressed("enter"):
    #             self.send_audio()
    #         else:
    #             time.sleep(1)

    # def thread3(self):
    #     keyboard.add_hotkey("q", self.send_audio)
    #     keyboard.add_hotkey("enter", self.send_audio)

    #     while self._run_thread_active:
    #         self.terminal_logger.info("Press Q to send audio", debug=True)
    #         time.sleep(0.1)  # Lower sleep time for better responsiveness

    # def thread(self, total_time, max_cpt, audio_data, chunk_size, sample_width, rate):
    #     while self._run_thread_active:
    #         # time.sleep(0.1)

    #         if keyboard.is_pressed("q"):
    #             self.terminal_logger.info("Q PRESSED")
    #             list_ius = []
    #             read_cpt = 0
    #             while read_cpt < max_cpt:
    #                 sample = audio_data[
    #                     (chunk_size * sample_width) * read_cpt : (chunk_size * sample_width) * (read_cpt + 1)
    #                 ]
    #                 read_cpt += 1
    #                 output_iu = self.create_iu(
    #                     audio=sample, raw_audio=sample, nframes=chunk_size, rate=rate, sample_width=sample_width
    #                 )
    #                 output_iu.dispatch = True
    #                 list_ius.append((output_iu, retico_core.UpdateType.ADD))

    #             # Add silence for VAD
    #             silence_duration = 1
    #             nb_silence_IUs = round(silence_duration / self.frame_length)
    #             print("nb_silence_IUs = ", nb_silence_IUs)
    #             for i in range(nb_silence_IUs):
    #                 silence_sample = b"\x00" * chunk_size * sample_width
    #                 output_iu = self.create_iu(
    #                     audio=silence_sample,
    #                     raw_audio=silence_sample,
    #                     nframes=rate * silence_duration,
    #                     rate=rate,
    #                     sample_width=sample_width,
    #                 )
    #                 output_iu.dispatch = True
    #                 list_ius.append((output_iu, retico_core.UpdateType.ADD))

    #             um = retico_core.UpdateMessage()
    #             um.add_ius(iu_list=list_ius)
    #             # time.sleep(total_time + silence_duration)
    #             self.append(um)
    #         else:
    #             self.terminal_logger.info("Q NOT PRESSED")
    #             time.sleep(0.1)

    def shutdown(self):
        super().shutdown()
        # self.p.terminate()
        self._run_thread_active = False
