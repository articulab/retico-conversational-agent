from functools import partial
from llama_cpp import Llama
import retico_core
import torch
import retico_conversational_agent as agent
from retico_core.log_utils import (
    filter_cases,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from dialogue_history_hf import DialogueHistoryHF

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
    "llama3.1_8B": [
        ["QuantFactory/Meta-Llama-3.1-8B-GGUF", "Meta-Llama-3.1-8B.Q4_K_M.gguf"],
        "meta-llama/Llama-3.1-8B",
    ],
    "llama3.1_8B_I": [
        ["QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"],
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
    "llama3.2_3B": [["QuantFactory/Llama-3.2-3B-GGUF", "Llama-3.2-3B.Q4_K_M.gguf"], "meta-llama/Llama-3.2-3B"],
    "llama3.2_3B_I": [
        ["QuantFactory/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct.Q4_K_M.gguf"],
        "meta-llama/Llama-3.2-3B-Instruct",
    ],
}
hf_token = ""
model = models["llama3.1_8B_I"]
# agent
# llama_cpp = Llama(
#     model_path=model_path,
#     n_ctx=context_size,
#     n_gpu_layers=n_gpu_layers,
#     verbose=verbose,
# )

tokenizer = Llama.from_pretrained(
    repo_id=model[0][0],
    filename=model[0][1],
    device_map=device,
    n_ctx=context_size,
    n_gpu_layers=n_gpu_layers,
    verbose=verbose,
)
model = tokenizer

# hf
dh_hf = DialogueHistoryHF(
    terminal_logger=terminal_logger,
    initial_dh=chat,
    context_size=context_size,
)
tokenizer_hf = AutoTokenizer.from_pretrained(model[1], token=hf_token)
# model_hf = AutoModelForCausalLM.from_pretrained(model[1], token=hf_token)


### PROMPTS
# print(len(llama_cpp.tokenize(system_prompt.encode("utf-8"))))
# dh = agent.DialogueHistory(
#     prompt_format_config,
#     terminal_logger=terminal_logger,
#     initial_dh=chat_2,
#     context_size=context_size,
# )
# prompt_dh, tokens = dh.prepare_dialogue_history(llama_cpp.tokenize)
# prompt_1_tokens = tokenizer_hf.apply_chat_template(
#     chat,
#     tokenize=True,
#     add_generation_prompt=False,
#     return_tensors="pt",
#     max_length=context_size,
#     truncation=True,
# )[0]
# prompt_1 = tokenizer_hf.decode(prompt_1_tokens)
prompt_2_tokens, dh = dh_hf.prepare_dialogue_history(tokenizer_hf.apply_chat_template)
prompt_2 = tokenizer_hf.decode(prompt_2_tokens)
# prints
# print(f"\n\nPrompt DH (nb tokens = {len(tokens)}): ", prompt_dh)
# print(f"\n\nPrompt 1 (nb tokens = {len(prompt_1_tokens)}) : ", prompt_1)
print(f"\n\nPrompt 2 (nb tokens = {len(prompt_2_tokens)}): ", prompt_2)


### Generation

# method one with generate
out_tokens = model.generate(prompt_2_tokens, max_new_tokens=100)
out_sentence = tokenizer_hf.decode(out_tokens[0])
out_sentence_2 = tokenizer.detokenize([out_tokens[0]]).decode("utf-8", errors="ignore")
print("\n\nOut sentence : ", out_sentence)
print("\n\nOut sentence 2 : ", out_sentence_2)
print("\n\nOut nb tokens : ", len(out_tokens))

# method 2 with chat completion
out_tokens = model.create_chat_completion(dh, max_new_tokens=100)
out_sentence = tokenizer_hf.decode(out_tokens[0])
out_sentence = tokenizer.detokenize([out_tokens[0]])
out_sentence_2 = tokenizer.detokenize([out_tokens[0]]).decode("utf-8", errors="ignore")
print("\n\nOut sentence : ", out_sentence)
print("\n\nOut sentence 2 : ", out_sentence_2)
print("\n\nOut nb tokens : ", len(out_tokens))
