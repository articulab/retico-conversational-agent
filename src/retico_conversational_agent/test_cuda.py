import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import retico_core
from retico_core import network
from retico_core.log_utils import (
    configurate_plot,
)

from .dialogue_history import DialogueHistory
from .LLM_DM import LlmDmModule
from .TTS_DM import TtsDmModule
from .ASR_DM import AsrDmModule


def test_cuda(module_names=["llm_local"], local_model=False):
    # parameters definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verbose = True
    log_folder = "logs/run"
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    model_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_name = "mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    rate = 16000
    tts_frame_length = 0.2
    tts_model = "jenny"
    system_prompt = b"This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"
    plot_config_path = "configs/plot_config_DM.json"
    prompt_format_config = "configs/prompt_format_config.json"
    context_size = 2000

    # configurate logger
    terminal_logger, _ = retico_core.log_utils.configurate_logger(log_folder)

    # configure plot
    configurate_plot(
        plot_config_path=plot_config_path,
    )

    modules = []

    print("device = ", device)
    print("module names = ", module_names)

    for module_name in module_names:
        if "llm" in module_name:
            print("llm init")
            dialogue_history = DialogueHistory(
                prompt_format_config,
                terminal_logger=terminal_logger,
                initial_system_prompt=system_prompt,
                context_size=context_size,
            )
            if module_name == "llm":
                llm = LlmDmModule(
                    None,
                    model_repo,
                    model_name,
                    dialogue_history=dialogue_history,
                    verbose=verbose,
                    device=device,
                )
                modules.append(llm)
            elif module_name == "llm_local":
                llm = LlmDmModule(
                    model_path,
                    None,
                    None,
                    dialogue_history=dialogue_history,
                    verbose=verbose,
                    device=device,
                )
                modules.append(llm)

        elif module_name == "tts":
            tts = TtsDmModule(
                language="en",
                model=tts_model,
                printing=False,
                frame_duration=tts_frame_length,
                device=device,
                verbose=verbose,
            )
            modules.append(tts)

        elif module_name == "asr":
            asr = AsrDmModule(
                device=device,
                full_sentences=True,
                input_framerate=rate,
                verbose=verbose,
            )
            modules.append(asr)

        if len(modules) > 1:
            modules[-1].subscribe(modules[-2])

    # running system
    try:
        m_list, _ = network.discover(modules[0])
        print("setup")
        for m in m_list:
            m.setup()
        print("stop")
        for m in m_list:
            m.stop()
    except Exception:
        print("exception")
        terminal_logger.exception()
        network.stop(modules[0])
