import numpy as np
import torch


def extract_token_durations_list(
    attentions,
    num_text_tokens,
    start_text_token,
    end_text_token,
    words_ids_list,
    hop_length=256,
    sample_rate=24000,
):

    alignments = []
    alignments_2 = []

    avg_scores_per_audio_token = get_avg_attention_score_per_audio_token(
        attentions, num_text_tokens, start_text_token, end_text_token
    )

    for j, words_ids in enumerate(words_ids_list):
        print(f"\n\n#########################\nwords_ids {j}")
        words_ids_alignments = []
        words_ids_alignments_2 = []
        for i, avg_scores in enumerate(avg_scores_per_audio_token):

            mean_att_per_word = [np.mean(avg_scores[words_ids[i][0] : words_ids[i][1]]) for i in range(len(words_ids))]

            words_ids_alignments.append(
                words_ids[np.argmax(mean_att_per_word)][1]
            )  # index of last token in the highest ranked word
            words_ids_alignments_2.append(np.argmax(mean_att_per_word))

            print(f"audio token {i}")
            print(f"avg_scores = {avg_scores}")
            print(f"mean_att_per_word = {np.array(mean_att_per_word)}\n")
            # alignments_2.append(np.argmax(avg_scores))  # index of aligned text token

        alignments.append(words_ids_alignments)
        alignments_2.append(words_ids_alignments_2)

    alignments = np.array(alignments)
    alignments_2 = np.array(alignments_2)
    print("alignments = ", alignments)
    print("alignments_2 = ", alignments_2)


def extract_token_alignment(attentions):
    """
    Extracts a [audio_token, text_token] alignment matrix from attention tuples.

    Args:
        attentions (tuple): GPT attention output, shaped as
            Tuple[Tuple[Tensor]], where each inner tensor has shape
            (1, num_heads, 1, num_text_tokens)

    Returns:
        alignment_matrix: torch.FloatTensor of shape [num_audio_tokens, num_text_tokens]
    """
    num_audio_tokens = len(attentions)
    num_layers = len(attentions[0])

    all_attns = []
    attn_vectors = []
    max_len = 0
    for t_idx in range(num_audio_tokens):  # loop over generated tokens
        # print("t_idx",t_idx)
        token_attns = []
        for l_idx in range(num_layers):  # loop over layers
            # print("l_idx",l_idx)
            attn = attentions[t_idx][l_idx]  # shape: [1, num_heads, 1, seq_len]
            if attn.dim() == 4:
                # Full attention matrix: [1, heads, tgt_len, src_len]
                if attn.size(2) > t_idx:
                    attn_for_token = attn[:, :, t_idx, :]  # slice on tgt_len dim
                else:
                    # Sometimes tgt_len = 1 but t_idx > 0, fallback to last token
                    attn_for_token = attn[:, :, -1, :]
            elif attn.dim() == 3:
                # Already only latest token attention: [1, heads, src_len]
                attn_for_token = attn
            else:
                raise ValueError(f"Unexpected attention tensor shape: {attn.shape}")

            attn_for_token = attn_for_token.squeeze(0)  # remove batch dim → [heads, src_len]
            attn_mean = attn_for_token.mean(dim=0)  # average over heads → [src_len]
            token_attns.append(attn_mean)
            # attn_for_token = attn[:, t_idx, :]  # attention of t_idx-th token over src tokens
            # # attn = attn.squeeze(0).squeeze(1)  # shape: [num_heads, seq_len]
            # # attn = attn.squeeze(0).squeeze(0)  # shape: [num_heads, seq_len]
            # print(f"Layer {l_idx} token {t_idx} attn shape: {attn_for_token.shape}")
            # attn_for_token = attn[:, :, t_idx, :]  # shape: [1, 1, src_len]
            # print(f"Layer {l_idx} token {t_idx} attn shape before squeeze: {attn_for_token.shape}")
            # attn_for_token = attn_for_token.squeeze(0).squeeze(0)  # shape: [src_len]
            # print(f"Layer {l_idx} token {t_idx} attn shape after squeeze: {attn_for_token.shape}")
            # token_attns.append(attn_for_token)
            # attn = attn_for_token.mean(dim=0)  # mean over heads → shape: [seq_len]
            # token_attns.append(attn)
        token_attn = torch.stack(token_attns).mean(dim=0)  # mean over layers
        # all_attns.append(token_attn)
        attn_vectors.append(token_attn)
        max_len = max(max_len, token_attn.size(0))

    # Second pass to pad all to same length
    # print("max_len = ", max_len)
    padded_attns = []
    for vec in attn_vectors:
        pad_len = max_len - vec.size(0)
        # print("vec shape = ", vec.shape, vec.size(0))
        # print("pad_len = ", pad_len)
        padded = torch.nn.functional.pad(vec, (0, pad_len), value=0.0)
        # print("padded vec shape = ", padded.shape, padded.size(0))
        padded_attns.append(padded)
    # print("return all_attns shape = ", len(padded_attns), [len(padded_attns[i]) for i in range(len(padded_attns))])
    time.sleep(30)

    # Final shape: [num_audio_tokens, num_text_tokens]
    # print("return all_attns shape = ", len(all_attns), [len(all_attns[i]) for i in range(len(all_attns))])
    # return torch.stack(all_attns)
    return torch.stack(padded_attns)


def compute_phoneme_timestamps(alignment_matrix, hop_time=1024 / 24000):
    """
    Computes start/end times for each text token.

    Args:
        alignment_matrix (Tensor): [num_audio_tokens, num_text_tokens]
        hop_time (float): Seconds per audio token

    Returns:
        List of (start_time, end_time) tuples per text token
    """
    audio_to_text = alignment_matrix.argmax(dim=1)  # audio_token_idx → text_token_idx
    text_token_to_audio = {}

    for audio_idx, text_idx in enumerate(audio_to_text.tolist()):
        if text_idx not in text_token_to_audio:
            text_token_to_audio[text_idx] = []
        text_token_to_audio[text_idx].append(audio_idx)

    token_times = []
    for idx in range(alignment_matrix.shape[1]):
        audio_indices = text_token_to_audio.get(idx, [])
        if not audio_indices:
            token_times.append((None, None))  # or fill with last valid time
        else:
            start = min(audio_indices) * hop_time
            end = (max(audio_indices) + 1) * hop_time
            token_times.append((start, end))

    return token_times


def get_avg_attention_score_per_audio_token(attentions, num_text_tokens, start_text_token, end_text_token):
    num_audio_tokens = len(attentions)  # i
    num_layers = len(attentions[0])  # 30
    num_heads = attentions[0][0].shape[1]  # 16
    print("num_text_tokens = ", num_text_tokens)
    print("num_audio_tokens = ", num_audio_tokens)
    print("num_layers = ", num_layers)
    print("num_heads = ", num_heads)

    avg_scores_per_audio_token = []

    for t in range(num_audio_tokens):
        # Average across layers and heads
        scores = []
        for layer in range(num_layers):
            attn = attentions[t][layer]  # Shape: [1, 16, 1, k_len]
            # k_len = gpt_cond_latent.shape[1] + bos + num_text_tokens + eos + start_audio_token ?
            attn = attn[:, :, 0, start_text_token:end_text_token]  # only text tokens
            attn_avg = attn.mean(dim=1).squeeze(0)  # Shape: [k_len]
            scores.append(attn_avg.cpu().numpy())

        avg_scores = np.mean(np.stack(scores), axis=0)  # Shape: [num_text_tokens]
        avg_scores_per_audio_token.append(avg_scores)
    return np.array(avg_scores_per_audio_token)


def extract_alignment(
    attentions,
    num_text_tokens,
    start_text_token,
    end_text_token,
    words_ids,
    alignment_method="mean_per_word",  # mean_per_word, argmax_word_id_text_token, argmax_text_token
):

    avg_scores_per_audio_token = get_avg_attention_score_per_audio_token(
        attentions, num_text_tokens, start_text_token, end_text_token
    )
    alignments = []
    for i, avg_scores in enumerate(avg_scores_per_audio_token):
        # take id of argmax text token
        if alignment_method == "argmax_text_token":
            alignments.append(np.argmax(avg_scores))
        # take id of word that contain the max text token
        elif alignment_method == "argmax_word_id_text_token":
            alignments.append(
                np.argmax([np.max(avg_scores[words_ids[i][0] : words_ids[i][1]]) for i in range(len(words_ids))])
            )
        # take id of word that has the max mean over text tokens
        elif alignment_method == "mean_per_word":
            mean_att_per_word = [np.mean(avg_scores[words_ids[i][0] : words_ids[i][1]]) for i in range(len(words_ids))]
            alignments.append(np.argmax(mean_att_per_word))

        print(f"audio token {i}")
        print(f"avg_scores = {avg_scores}")
        print(f"mean_att_per_word = {np.array(mean_att_per_word)}\n")

    alignments = np.array(alignments)
    print("alignments = ", alignments)
    return alignments


def extract_token_durations(
    attentions,
    num_text_tokens,
    start_text_token,
    end_text_token,
    words_ids,
    hop_length=256,
    sample_rate=24000,
    nb_frames_per_audio_token=1024 * 24000 / 22050,  # upscaling
):

    alignments = extract_alignment(attentions, num_text_tokens, start_text_token, end_text_token, words_ids)

    # Replace alignments to have only increasing words_ids in alignments
    for i in range(len(alignments) - 1):
        if alignments[i + 1] < alignments[i]:
            alignments[i + 1] = alignments[i]
    print("alignments = ", alignments)

    # Count words durations in number of associated audio tokens
    words_durations_in_nb_audio_token = np.zeros(words_ids.shape[0])
    for idx in alignments:
        words_durations_in_nb_audio_token[idx] += 1
    print("durations = ", words_durations_in_nb_audio_token)

    # Count words durations in nb associated audio frames (converting audio token to frames)
    # nb_frames = nb_audio_token * gpt_code_stride_len * output_framerate * input_framerate = nb_audio_token * 1024 * 24000 / 22050 # upscaling
    words_durations_in_nb_frames = words_durations_in_nb_audio_token * nb_frames_per_audio_token

    # Convert token counts to time durations (in seconds)
    time_per_token = hop_length / sample_rate  # 0.01067 seconds
    words_durations_in_sec = words_durations_in_nb_audio_token * time_per_token
    print("durations_sec = ", words_durations_in_sec)

    return (
        words_durations_in_nb_frames,
        words_durations_in_sec,
        alignments,
    )  # durations per text token, and raw alignment map


def get_dur(gpt_cond_latent_shape, text_tokens, att_attentions, detokenized_text_tokens, space_token=2):
    print("text tokens : ", text_tokens)
    print("gpt_cond_latent shape : ", gpt_cond_latent_shape)
    print("detokenized text : ", detokenized_text_tokens)
    print("space token : ", space_token)
    # assert space_token == 2
    space_token_ids = np.where(text_tokens[0].cpu() == space_token)[0]
    print("space_token_ids : ", space_token_ids)

    if space_token_ids.shape[0] == 0:
        print("No space token found in text tokens.")
        words_ids = np.array([[0, text_tokens.shape[-1]]])
    else:
        words_ids_list = []
        # method 1 : no lan
        words_ids = np.concatenate(
            (
                [[1, space_token_ids[0]]],
                [[space_token_ids[i], space_token_ids[i + 1]] for i in range(len(space_token_ids) - 1)],
                [[space_token_ids[-1], text_tokens.shape[-1]]],
            )
        )
        words_ids_list.append(words_ids)
        print("words_ids : ", words_ids)
        print(
            "words_ids words : ",
            [detokenized_text_tokens[words_ids[i][0] : words_ids[i][1]] for i in range(len(words_ids))],
        )

        # # method 2 : no lan, no punctuation, words include spaces
        # words_ids = np.concatenate(
        #     (
        #         [[1, space_token_ids[0]]],
        #         [[space_token_ids[i], space_token_ids[i + 1]] for i in range(len(space_token_ids) - 1)],
        #         [[space_token_ids[-1], text_tokens.shape[-1] - 1]],
        #     )
        # )
        # words_ids_list.append(words_ids)
        # print("words_ids : ", words_ids)
        # print("words_ids words : ", [detokenized_text_tokens[words_ids[i][0] : words_ids[i][1]] for i in range(len(words_ids))])

        # # method 3 : no lan, no punctuation, words does not include spaces
        # words_ids = np.concatenate(
        #     (
        #         [[1, space_token_ids[0]]],
        #         [[space_token_ids[i] + 1, space_token_ids[i + 1]] for i in range(len(space_token_ids) - 1)],
        #         [[space_token_ids[-1] + 1, text_tokens.shape[-1] - 1]],
        #     )
        # )
        # words_ids_list.append(words_ids)
        # print("words_ids : ", words_ids)
        # print("words_ids words : ", [detokenized_text_tokens[words_ids[i][0] : words_ids[i][1]] for i in range(len(words_ids))])

    # k_len = gpt_cond_latent_shape[1] + bos + num_text_tokens + eos + start_audio_token ?
    start_text_token = gpt_cond_latent_shape[1] + 1
    end_text_token = gpt_cond_latent_shape[1] + 1 + len(text_tokens[0])
    print("start_text_token = ", start_text_token)
    print("end_text_token = ", end_text_token)
    words_durations_in_nb_frames, words_durations_in_sec, alignments = extract_token_durations(
        attentions=att_attentions,
        num_text_tokens=len(text_tokens[0]),
        start_text_token=start_text_token,
        end_text_token=end_text_token,
        words_ids=words_ids,
        hop_length=256,
        sample_rate=24000,
    )
    # print("att_alignment shape = ", att_alignment.shape)

    # # print("att shape = ", att_attentions.shape)
    # att_alignment = extract_token_alignment(att_attentions)
    # print("att_alignment shape = ", att_alignment.shape)
    # # print("att_alignment = ", att_alignment)
    # token_timestamps = compute_phoneme_timestamps(att_alignment)
    # print("token_timestamps = ", token_timestamps)
    # # time.sleep(300)
    # # print("att type = ", type(att_attentions))
    # # print("att shape = ", att_attentions[0].shape)
    # # attn_tensor = torch.stack(att_attentions)
    # # avg_attn = attn_tensor.mean(dim=(0, 1, 2))
    # # alignment = avg_attn.argmax(dim=1)
    # # print("alignment shape = ", alignment.shape)
    # # print("alignment = ", alignment)
    # # gpt_codes.shape[-1] = N tokens
    # # durations.append(gpt_codes.shape[-1])
    # durations = torch.tensor([1]*gpt_codes.shape[-1])
    # durations_sens = durations.reshape(1, durations.shape[0]) if durations_sens is None else torch.stack([durations_sens, durations])
    # gpt_codes_sens = gpt_codes if gpt_codes_sens is None else torch.stack([gpt_codes_sens, gpt_codes])
    # print("NB GENERATED TOKENS XTTS", gpt_codes.shape[-1])
    # print("audio len generated = ", gpt_codes.shape[-1]*(1024 / 24000))
    return words_durations_in_nb_frames, words_durations_in_sec, alignments


def get_words_durations(alignment_required_data):
    """
    Get the words durations from the alignment required data.

    Args:
        alignment_required_data (dict): The alignment required data.

    Returns:
        tuple: The words durations, the durations length, and the alignment.
    """
    # Extract the relevant data from the dictionary
    att_attentions = alignment_required_data["att_attentions"]
    gpt_cond_latent_shape = alignment_required_data["gpt_cond_latent_shape"]
    text_tokens = alignment_required_data["text_tokens"]
    # language = alignment_required_data["language"]
    detokenized_text_tokens = alignment_required_data["detokenized_text_tokens"]

    # Call the function to get the words durations
    return get_dur(gpt_cond_latent_shape, text_tokens, att_attentions, detokenized_text_tokens)
