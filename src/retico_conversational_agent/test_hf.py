from functools import partial
from llama_cpp import Llama
import retico_core
import torch
import retico_conversational_agent as agent
from retico_core.log_utils import (
    filter_cases,
)
from transformers import AutoTokenizer

from retico_conversational_agent.dialogue_history_hf import DialogueHistoryHF

# parameters
filters = [
    partial(
        filter_cases,
        cases=[
            [("debug", [True])],
            # [("module", ["VAD DM Module"])],
            [("level", ["warning", "error"])],
        ],
    )
]
log_folder = "logs/run"
terminal_logger, _ = retico_core.log_utils.configurate_logger(log_folder, filters=filters)
prompt_format_config = "configs/prompt_format_config.json"
# context_size = 2048
context_size = 250
n_gpu_layers = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
verbose = False
system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. The teacher is teaching mathemathics to the child student. As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation. You play the role of a teacher. Here is the beginning of the conversation :"
chat = [
    {"role": "system", "content": system_prompt, "turn_id": -1},
    {"role": "user", "content": "Hello, how are you?", "turn_id": 0},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 1},
    {"role": "user", "content": "I'd like to show off how chat templating works!", "turn_id": 2},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 3},
    {"role": "user", "content": "I'd like to show off how chat templating works!", "turn_id": 4},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 5},
    {"role": "user", "content": "I'd like to show off how chat templating works!", "turn_id": 6},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 7},
    {"role": "user", "content": "I'd like to show off how chat templating works!", "turn_id": 8},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 9},
    {"role": "user", "content": "I'd like to show off how chat templating works!", "turn_id": 10},
]
chat_2 = [c.copy() for c in chat]
for idc, c in enumerate(chat_2):
    c["text"] = c["content"]
    c["speaker"] = "agent" if c["role"] == "assistant" else "user"
    del c["content"]
    del c["role"]
    c["turn_id"] = idc
chat_2[0]["speaker"] = "system_prompt"
chat_2[0]["turn_id"] = -1

print("\n\nChat : ", chat)
print("\n\nChat 2 : ", chat_2)

models = {
    "mistral_inst": [
        ["TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"],
        "mistralai/Mistral-7B-Instruct-v0.2",
    ],
    "llama2_7B": [["TheBloke/Llama-2-7b-Chat-GGUF", "llama-2-7b-chat.q4_K_M.gguf"], "meta-llama/Llama-2-7b-chat-hf"],
}
hf_token = "secret"
model = models["llama2_7B"]
# agent
# llama_cpp = Llama(
#     model_path=model_path,
#     n_ctx=context_size,
#     n_gpu_layers=n_gpu_layers,
#     verbose=verbose,
# )

llama_cpp = Llama.from_pretrained(
    repo_id=model[0][0],
    filename=model[0][1],
    device_map=device,
    n_ctx=context_size,
    n_gpu_layers=n_gpu_layers,
    verbose=verbose,
)

print(len(llama_cpp.tokenize(system_prompt.encode("utf-8"))))


dh = agent.DialogueHistory(
    prompt_format_config,
    terminal_logger=terminal_logger,
    initial_dh=chat_2,
    context_size=context_size,
)

prompt_dh, tokens = dh.prepare_dialogue_history(llama_cpp.tokenize)

# hf
dh_hf = DialogueHistoryHF(
    terminal_logger=terminal_logger,
    initial_dh=chat,
    context_size=context_size,
)
tokenizer = AutoTokenizer.from_pretrained(model[1], token=hf_token)
tokenized_prompt_1 = tokenizer.apply_chat_template(
    chat,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    max_length=context_size,
    truncation=True,
)
prompt_1 = tokenizer.decode(tokenized_prompt_1[0])
prompt_2_tokens = dh_hf.prepare_dialogue_history(tokenizer.apply_chat_template)
prompt_2 = tokenizer.decode(prompt_2_tokens)
# prints
print("\n\nPrompt DH : ", prompt_dh)
print("\n\nPrompt 1 : ", prompt_1)
print("\n\nPrompt 2 : ", prompt_2)
