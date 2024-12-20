import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import retico_core
from retico_core import network, audio
from functools import partial
import torch


from simple_retico_agent.dialogue_history import DialogueHistory
from simple_retico_agent.simple_vad import SimpleVADModule
from simple_retico_agent.simple_whisper_asr import SimpleWhisperASRModule
from simple_retico_agent.simple_llm import SimpleLLMModule
from simple_retico_agent.simple_tts import SimpleTTSModule
from simple_retico_agent.simple_speaker import SimpleSpeakerModule

from retico_conversational_agent.dialogue_history import DialogueHistory
from retico_conversational_agent.VAD_DM import VadModule
from retico_conversational_agent.ASR_DM import AsrDmModule
from retico_conversational_agent.LLM_DM import LlmDmModule
from retico_conversational_agent.TTS_DM import TtsDmModule
from retico_conversational_agent.dialogue_manager import (
    DialogueManagerModule,
    DialogueManagerModule_2,
)


from retico_core.log_utils import (
    filter_has_key,
    filter_does_not_have_key,
    filter_value_in_list,
    filter_value_not_in_list,
    filter_conditions,
    filter_cases,
    configurate_plot,
    plot_once,
)


def main_simple():
    # parameters definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_folder = "logs/run"
    frame_length = 0.02
    tts_frame_length = 0.2
    rate = 16000
    tts_model_samplerate = 48000
    # model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"
    plot_config_path = "configs/plot_config_simple.json"
    plot_live = True
    prompt_format_config = "configs/prompt_format_config.json"
    context_size = 2000

    # filters
    filters = [
        partial(
            filter_cases,
            cases=[
                [("debug", [True])],
                [("level", ["warning", "error"])],
            ],
        )
    ]
    # filters = []
    # configurate logger
    terminal_logger, _ = retico_core.log_utils.configurate_logger(
        log_folder, filters=filters
    )

    # configure plot
    configurate_plot(
        is_plot_live=plot_live,
        refreshing_time=1,
        plot_config_path=plot_config_path,
        window_duration=30,
    )

    dialogue_history = DialogueHistory(
        prompt_format_config,
        terminal_logger=terminal_logger,
        initial_system_prompt=system_prompt,
        context_size=context_size,
    )

    # create modules
    mic = audio.MicrophoneModule()

    vad = SimpleVADModule(
        input_framerate=rate,
        frame_length=frame_length,
    )

    asr = SimpleWhisperASRModule(device=device, framerate=rate)

    llm = SimpleLLMModule(
        model_path=None,
        model_repo=model_repo,
        model_name=model_name,
        dialogue_history=dialogue_history,
        device=device,
        context_size=context_size,
        # verbose=True,
    )

    tts = SimpleTTSModule(
        frame_duration=tts_frame_length,
        device=device,
    )

    speaker = SimpleSpeakerModule(rate=tts_model_samplerate)

    # create network
    mic.subscribe(vad)
    vad.subscribe(asr)
    asr.subscribe(llm)
    llm.subscribe(tts)
    tts.subscribe(speaker)
    speaker.subscribe(vad)

    # running system
    try:
        network.run(mic)
        print("Dialog system running until ENTER key is pressed")
        input()
        network.stop(mic)
    except Exception:
        terminal_logger.exception("exception in main")
        network.stop(mic)
    finally:
        plot_once(plot_config_path=plot_config_path)


def main_DM():
    """The `main_DM` function creates and runs a dialog system that is able to
    have a conversation with the user.

    The dialog system is composed of different modules: - a Microphone :
    captures the user's voice - an ASR : transcribes the user's voice
    into text - a LLM : generates a textual answer to the trancription
    from user's spoken sentence. - a TTS : generates a spoken answer
    from the LLM's textual answer. - a Speaker : outputs the spoken
    answer generated by the system.

    We provide the system with a scenario (contained in the
    "system_prompt") that it will follow through the conversation : The
    system is a teacher and it will teach mathematics to a 8-year-old
    child student (the user)

    the parameters defined : - model_path : the path to the weights of
    the LLM that will be used in the dialog system. - system_prompt : a
    part of the prompt that will be given to the LLM at every agent turn
    to set the scenario of the conversation. - printing : an argument
    that set to True will print a lot of information useful for
    degugging. - rate : the target audio signal rate to which the audio
    captured by the microphone will be converted to (so that it is
    suitable for every module) - frame_length : the chosen frame length
    in seconds at which the audio signal will be chunked. - log_folder :
    the path to the folder where the logs (information about each
    module's latency) will be saved.

    It is recommended to not modify the rate and frame_length parameters
    because the modules were coded with theses values and it is not
    ensured that the system will run correctly with other values.
    """

    # parameters definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    printing = False
    log_folder = "logs/run"
    frame_length = 0.02
    tts_frame_length = 0.2
    # tts_frame_length = 0.02
    rate = 16000
    # tts_model_samplerate = 22050
    # tts_model = "vits_vctk"
    tts_model_samplerate = 48000
    tts_model = "jenny"
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"
    plot_config_path = "configs/plot_config_3.json"
    plot_live = True
    module_order = [
        "Microphone",
        "VAD",
        "DialogueManager",
        "WhisperASR",
        "LLM",
        "TTS",
        "Speaker",
    ]
    prompt_format_config = "configs/prompt_format_config.json"
    context_size = 2000

    # filters
    filters = [
        partial(
            filter_cases,
            cases=[
                [("debug", [True])],
                # [("debug", [True]), ("module", ["DialogueManager Module"])],
                [("level", ["warning", "error"])],
            ],
            # cases=[
            #     [("module", ["DialogueManager Module"])],
            #     [("level", ["warning", "error"])],
            # ],
        )
    ]
    # filters = []
    # configurate logger
    terminal_logger, _ = retico_core.log_utils.configurate_logger(
        log_folder, filters=filters
    )

    # configurate logger
    # terminal_logger, _ = retico_core.log_utils.configurate_logger(log_folder)

    # configure plot
    configurate_plot(
        is_plot_live=plot_live,
        refreshing_time=1,
        plot_config_path=plot_config_path,
        module_order=module_order,
        window_duration=30,
    )

    dialogue_history = DialogueHistory(
        prompt_format_config,
        terminal_logger=terminal_logger,
        initial_system_prompt=system_prompt,
        context_size=context_size,
    )

    # create modules
    # mic = MicrophonePTTModule(rate=rate, frame_length=frame_length)
    # mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)
    mic = audio.MicrophoneModule()

    vad = VadModule(
        input_framerate=rate,
        frame_length=frame_length,
    )

    dm = DialogueManagerModule(
        dialogue_history=dialogue_history,
        input_framerate=rate,
        frame_length=frame_length,
    )
    # dm = DialogueManagerModule_2(
    #     dialogue_history=dialogue_history,
    #     input_framerate=rate,
    #     frame_length=frame_length,
    # )
    dm.add_repeat_policy()
    dm.add_soft_interruption_policy()
    dm.add_continue_policy()
    dm.add_backchannel_policy()

    # asr = WhisperASRInterruptionModule(
    #     device=device,
    #     printing=printing,
    #     full_sentences=True,
    #     input_framerate=rate,
    # )

    asr = AsrDmModule(
        device=device,
        full_sentences=True,
        input_framerate=rate,
    )

    # llm = LlamaCppMemoryIncrementalInterruptionModule(
    #     model_path,
    #     None,
    #     None,
    #     None,
    #     system_prompt,
    #     printing=printing,
    #     device=device,
    #     context_size=context_size,
    # )

    llm = LlmDmModule(
        model_path,
        None,
        None,
        dialogue_history=dialogue_history,
        printing=printing,
        device=device,
    )

    # tts = CoquiTTSInterruptionModule(
    #     language="en",
    #     model=tts_model,
    #     printing=printing,
    #     frame_duration=tts_frame_length,
    #     device=device,
    # )

    tts = TtsDmModule(
        language="en",
        model=tts_model,
        printing=printing,
        frame_duration=tts_frame_length,
        device=device,
    )

    speaker = SpeakerInterruptionModule(
        rate=tts_model_samplerate,
    )

    # vad = VADTurnModule(
    #     printing=printing,
    #     input_framerate=rate,
    #     frame_length=frame_length,
    # )

    # create network
    mic.subscribe(vad)
    vad.subscribe(dm)
    dm.subscribe(asr)
    dm.subscribe(llm)
    dm.subscribe(tts)
    dm.subscribe(speaker)
    asr.subscribe(llm)
    llm.subscribe(tts)
    tts.subscribe(speaker)
    speaker.subscribe(llm)
    speaker.subscribe(vad)
    speaker.subscribe(dm)

    # running system
    try:
        network.run(mic)
        # terminal_logger.info("Dialog system running until ENTER key is pressed")
        print("Dialog system running until ENTER key is pressed")
        input()
        network.stop(mic)
    except Exception:
        terminal_logger.exception("exception in main")
        network.stop(mic)
    finally:
        plot_once(
            plot_config_path=plot_config_path,
            module_order=module_order,
        )


import numpy as np

if __name__ == "__main__":
    # main_simple()
    main_DM()

    # plot_once(plot_config_path="configs/plot_config_simple.json")
