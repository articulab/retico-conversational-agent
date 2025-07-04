"""
Whisper ASR Module
==================

A retico module that provides Automatic Speech Recognition (ASR) using a
OpenAI's Whisper model. Periodically predicts a new text hypothesis from
the input incremental speech and predicts a final hypothesis when it is
the user end of turn.

The received VADTurnAudioIU are stored in a buffer from which a
prediction is made periodically, the words that were not present in the
previous hypothesis are ADDED, in contrary, the words that were present,
but aren't anymore are REVOKED. It recognize the user's EOT information
when COMMIT VADTurnAudioIUs are received, a final prediciton is then
made and the corresponding IUs are COMMITED.

The faster_whisper library is used to speed up the whisper inference.

Inputs : VADTurnAudioIU

Outputs : SpeechRecognitionIU
"""

import os
import threading
import time
from typing import Tuple
import numpy as np
import transformers
from faster_whisper import WhisperModel

import retico_core
from retico_core.log_utils import log_exception


from .utils import device_definition
from .additional_IUs import DMIU, SpeechRecognitionTurnIU

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AbstractASRSubclass:
    @staticmethod
    def setup(self):
        """Setup function that initialize the ASR model.

        Raises:
            NotImplementedError: Raises NotImplementedError if not implemented in subclass.
        """
        raise NotImplementedError()

    @staticmethod
    def produce(
        self,
        full_audio: bytes,
    ) -> str:
        """Subclass's producing function that calls the ASR and handles all ASR-related
        pre and post-processing to return a formatted result.

        Args:
            full_audio (bytes): The audio bytes to transcribe.

        Raises:
            NotImplementedError: Raises NotImplementedError if not implemented in subclass.

        Returns:
            str: Returns the audio's transcription.
        """
        raise NotImplementedError()


class WhisperASRSubclass(AbstractASRSubclass):
    def __init__(
        self,
        whisper_model="distil-large-v2",
        device=None,
        **kwargs,
    ):
        """Initializes the WhisperASRInterruption Module.

        Args:
            whisper_model (string): name of the desired model, has to
                correspond to a model in the faster_whisper library.
            device (string): wether the model will be executed on cpu or
                gpu (using "cuda").
        """
        self.device = device_definition(device)
        self.whisper_model = whisper_model

    def setup(self):
        self.model = WhisperModel(self.whisper_model, device=self.device, compute_type="int8")

    def produce(self, full_audio: bytes) -> str:
        audio_np = retico_core.audio.convert_audio_PCM16_to_float32(raw_audio=full_audio)
        segments, _ = self.model.transcribe(audio_np)  # the segments can be streamed
        segments = list(segments)
        transcription = "".join([s.text for s in segments])
        return transcription


class AsrDmModuleSubclassFaster(retico_core.AbstractModule):
    """A retico module that provides Automatic Speech Recognition (ASR) using a
    OpenAI's Whisper model. Periodically predicts a new text hypothesis from
    the input incremental speech and predicts a final hypothesis when it is the
    user end of turn.

    The received VADTurnAudioIU are stored in a buffer from which a
    prediction is made periodically, the words that were not present in
    the previous hypothesis are ADDED, in contrary, the words that were
    present, but aren't anymore are REVOKED. It recognize the user's EOT
    information when COMMIT VADTurnAudioIUs are received, a final
    prediciton is then made and the corresponding IUs are COMMITED.

    The faster_whisper library is used to speed up the whisper
    inference.

    Inputs : VADTurnAudioIU

    Outputs : SpeechRecognitionTurnIU
    """

    @staticmethod
    def name():
        return "ASR Whisper DM Module"

    @staticmethod
    def description():
        return "A module that recognizes transcriptions from speech using Whisper."

    @staticmethod
    def input_ius():
        return [DMIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionTurnIU

    def __init__(
        self,
        device=None,
        input_framerate=16000,
        subclass: AbstractASRSubclass = WhisperASRSubclass,
        **kwargs,
    ):
        """Initializes the WhisperASRInterruption Module.

        Args:
            device (string): wether the model will be executed on cpu or
                gpu (using "cuda").
            input_framerate (int, optional): input_framerate of the received VADIUs.
                Defaults to 16000.
        """
        super().__init__(**kwargs)
        self.subclass = subclass(device=device, **kwargs)

        self._asr_thread_active = False
        self.latest_input_iu = None
        self.audio_buffer = []
        self.input_framerate = input_framerate
        self.send_transcription = False
        self.process_audio = False
        self.transcription_um = None

    def process_update(self, update_message):
        """Receives and stores the audio from the DMIUs in the
        self.audio_buffer buffer.

        Args:
            update_message (UpdateType): UpdateMessage that contains new
                IUs, if their UpdateType is ADD, they are added to the
                audio_buffer.
        """
        process_audio = False
        for iu, ut in update_message:
            if iu.action == "process_audio":
                if self.input_framerate != iu.rate:
                    raise ValueError("input_framerate differs from iu input_framerate")
                # ADD corresponds to new audio chunks of user sentence, to generate new transcription hypothesis
                if ut == retico_core.UpdateType.COMMIT or ut == retico_core.UpdateType.ADD:
                    process_audio = True
                    self.audio_buffer.append(iu.raw_audio)
                    if not self.latest_input_iu:
                        self.latest_input_iu = iu
            elif iu.action == "send_transcription":
                self.send_transcription = True
        if process_audio:
            self.terminal_logger.debug("start_answer_generation", cl="trace")
            self.file_logger.info("start_answer_generation", last_iu_iuid=update_message._msgs[-1][0].iuid)
            self.process_audio = process_audio

    def _asr_thread(self):
        """Function used as a thread in the prepare_run function. Handles the
        messaging aspect of the retico module. Calls the Whisper model to
        generate a prediction from the audio contained in the audio_buffer
        sub-class's. ADD the new words and COMMITS the final prediction.

        (Only called at the user EOT for now).
        """
        while self._asr_thread_active:
            try:
                if self.process_audio:
                    self.file_logger.info("before predict")
                    self.terminal_logger.debug("process_audio")
                    full_audio = b"".join(self.audio_buffer)
                    prediction = self.subclass.produce(full_audio)
                    self.file_logger.info("after predict")
                    if len(prediction) != 0:
                        um, new_tokens = retico_core.text.get_text_increment(self, prediction)
                        for i, token in enumerate(new_tokens):
                            output_iu = self.create_iu(
                                grounded_in=self.latest_input_iu,
                                predictions=[prediction],
                                text=token,
                                stability=0.0,
                                confidence=0.99,
                                final=self.process_audio and (i == (len(new_tokens) - 1)),
                                # final=True,
                                turn_id=self.latest_input_iu.turn_id,
                            )
                            um.add_iu(output_iu, retico_core.UpdateType.ADD)

                            um.add_iu(output_iu, retico_core.UpdateType.COMMIT)
                            self.transcription_um = um
                            self.process_audio = False
                            self.terminal_logger.debug("end process_audio")

                elif self.send_transcription:
                    self.terminal_logger.debug("send_transcription")
                    self.append(self.transcription_um)
                    for iu, ut in self.transcription_um:
                        self.commit(iu)
                    self.audio_buffer = []
                    self.send_transcription = False
                    self.latest_input_iu = None
                    self.transcription_um = None
                    self.terminal_logger.debug("end send_transcription")
                    self.file_logger.info("send_clause")

                else:
                    time.sleep(0.01)

            except Exception as e:
                log_exception(module=self, exception=e)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.subclass.setup()

    def prepare_run(self):
        super().prepare_run()
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        super().shutdown()
        self._asr_thread_active = False
