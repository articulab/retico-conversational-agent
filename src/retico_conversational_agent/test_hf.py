from functools import partial
from llama_cpp import Llama, llama_chat_format
import retico_core
import torch
import retico_conversational_agent as agent
from retico_core.log_utils import (
    filter_cases,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from dialogue_history_hf import DialogueHistoryHf

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
context_size = 450
dh_size = 250
# n_gpu_layers = 100
n_gpu_layers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
verbose = True
system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. The teacher is teaching mathemathics to the child student. As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation. You play the role of a teacher. Here is the beginning of the conversation :"
chat = [
    {"role": "system", "content": system_prompt, "turn_id": -1},
    {"role": "user", "content": "Hello, how are you?", "turn_id": 0},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 1},
    {"role": "user", "content": "5 I'd like to show off how chat templating works!", "turn_id": 2},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 3},
    {"role": "user", "content": "4 I'd like to show off how chat templating works!", "turn_id": 4},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 5},
    {"role": "user", "content": "3 I'd like to show off how chat templating works!", "turn_id": 6},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 7},
    {"role": "user", "content": "2 I'd like to show off how chat templating works!", "turn_id": 8},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?", "turn_id": 9},
    {"role": "user", "content": "1 I'd like to show off how chat templating works!", "turn_id": 10},
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
generator = tokenizer
formatter = (
    (
        generator.chat_handler
        or generator._chat_handlers.get(generator.chat_format)
        or llama_chat_format.get_chat_completion_handler(generator.chat_format)
    )
    .__closure__[0]
    .cell_contents
)


def apply_chat_template_f(messages, model, formatter):
    result = formatter(messages=messages)
    prompt = model.tokenize(
        result.prompt.encode("utf-8"),
        add_bos=not result.added_special,
        special=True,
    )
    return prompt


# hf
dh_hf = DialogueHistoryHf(
    terminal_logger=terminal_logger,
    initial_dh=chat,
    context_size=dh_size,
)
tokenizer_hf = AutoTokenizer.from_pretrained(model[1], token=hf_token)

apply_chat_template_2 = partial(apply_chat_template_f, model=generator, formatter=formatter)
apply_chat_template_3 = partial(tokenizer_hf.apply_chat_template, tokenize=True, add_generation_prompt=True)
# generator_hf = AutoModelForCausalLM.from_pretrained(model[1], token=hf_token)
print("chat formatted", apply_chat_template_2(chat))
print("chat formatted", apply_chat_template_3(chat))

### PROMPTS

prompt_2_tokens, history = dh_hf.prepare_dialogue_history(apply_chat_template_2)
prompt_2 = tokenizer_hf.decode(prompt_2_tokens)
print(f"\n\nPrompt 2 (nb tokens = {len(prompt_2_tokens)}): ", prompt_2)
prompt_2_tokens, history = dh_hf.prepare_dialogue_history(apply_chat_template_3)
prompt_2 = tokenizer_hf.decode(prompt_2_tokens)
print(f"\n\nPrompt 3 (nb tokens = {len(prompt_2_tokens)}): ", prompt_2)

# ### Generation

# # method 1 with generate
# print("EOS TOKENS", generator.token_eos())


# def stop_function(tokens, logits):
#     return tokens[-1] == generator.token_eos()


# out_tokens = []
# try:
#     for token in generator.generate(prompt_2_tokens, stopping_criteria=stop_function):
#         out_tokens.append(token)
#         # out_tokens = torch.stack(out_tokens).squeeze(0).tolist()
#         if len(out_tokens) >= 100:
#             break
# except Exception as e:
#     print("Exception : ", e)

# out_sentence = tokenizer_hf.decode(out_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# out_sentence_2 = tokenizer.detokenize(out_tokens).decode("utf-8", errors="ignore")
# print("\n\nOut sentence : ", out_sentence)
# print("\n\nOut sentence 2 : ", out_sentence_2)
# print("\n\nOut nb tokens : ", len(out_tokens))

# # method 2 with chat completion
# result = generator.create_chat_completion(history, max_tokens=100)
# print("\n\nOut tokens : ", result)
# nb_tokens = result["usage"]["completion_tokens"]
# message = result["choices"][0]["message"]
# stop_reason = result["choices"][0]["finish_reason"]
# print("\n\nOut sentence : ", message)
# print("\n\nstop_reason : ", stop_reason)
# print("\n\nOut nb tokens : ", nb_tokens)
