from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch

# Load model and tokenizer
model_name = "gpt2"
# model_path = "mistralai/Mistral-7B-Instruct-v0.2"
# device = "auto"
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map=device,
#     quantization_config=quantization_config,
# )
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(tokenizer(".", add_special_tokens=False).input_ids)
print(tokenizer(" .", add_special_tokens=False).input_ids)
print(tokenizer(" . ", add_special_tokens=False).input_ids)
print(tokenizer("My dog, which", add_special_tokens=False).input_ids)
print(tokenizer("My dog.", add_special_tokens=False).input_ids)
print(tokenizer("My dog. ", add_special_tokens=False).input_ids)


class ReplacePhraseStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, pattern_text, replacement_text):
        self.tokenizer = tokenizer
        self.pattern_tokens = tokenizer(pattern_text, add_special_tokens=False).input_ids
        self.replacement_tokens = tokenizer(replacement_text, add_special_tokens=False).input_ids
        print(f"pattern to replace : {self.pattern_tokens}")
        print(f"replacement pattern : {self.replacement_tokens}")

    def __call__(self, input_ids, scores, **kwargs):
        """
        This function runs during `generate()`, checks if the unwanted phrase is generated,
        and modifies the generated sequence accordingly.
        """
        # Check each sequence in the batch
        for batch_idx in range(input_ids.shape[0]):
            generated_sequence = input_ids[batch_idx].tolist()
            # If the sequence contains the unwanted phrase, replace it
            if (
                len(generated_sequence) >= len(self.pattern_tokens)
                and generated_sequence[-len(self.pattern_tokens) :] == self.pattern_tokens
            ):
                print("\n\nreplacement pattern hit\n\n")
                # Replace the unwanted phrase
                modified_sequence = generated_sequence[: -len(self.pattern_tokens)] + self.replacement_tokens
                print(f"old sequence {generated_sequence}")
                print(f"new sequence {modified_sequence}")
                print(f"old input_ids {input_ids}")

                # Update the input_ids with the modified sequence
                input_ids[batch_idx, : len(modified_sequence)] = torch.tensor(
                    modified_sequence, device=input_ids.device
                )

                print(f"new input_ids {input_ids}")

                # Shorten the sequence if necessary (since we replaced multiple tokens with a single one)
                new_length = len(modified_sequence)
                input_ids = input_ids[:, :new_length]

        return False  # Do not stop generation, just modify it


# Create the stopping criteria object
# replace_criteria = ReplacePhraseStoppingCriteria(tokenizer, ", which", ".")
replace_criteria = ReplacePhraseStoppingCriteria(tokenizer, ", who", ". ")

# Example prompt
prompt = "My dog"

# Tokenize input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text with the stopping criteria applied
output_ids = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,  # Beam search enabled
    stopping_criteria=StoppingCriteriaList([replace_criteria]),
)
print(f"replacement pattern : {output_ids}")

# Decode and print the output
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
