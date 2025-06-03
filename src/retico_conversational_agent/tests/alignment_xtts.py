import numpy as np


def extract_words_durations(
    attentions,
    num_text_tokens,
    start_text_token,
    end_text_token,
    words_ids,
    internal_sample_rate=22050,
    output_sample_rate=24000,
    gpt_code_stride_len=1024,
    # nb_frames_per_audio_token=1024 * 24000 / 22050,  # upscaling
    alignment_method="mean_per_word",  # mean_per_word, argmax_word_id_text_token, argmax_text_token
    strict_increase=True,
):
    """Extract the words durations in number of frames and seconds, by calculating, from attention,
        the alignment between audio tokens and the text tokens.

    Args:
        attentions (torch.tensor): The attentions tensor.
        num_text_tokens (int): number of text tokens.
        start_text_token (int): the index of the start text token.
        end_text_token (int): the index of the end text token.
        words_ids (list): list of starting and ending ids of each word in the text token list.
        sample_rate (int, optional): sample rate of outputted audio. Defaults to 24000.
        nb_frames_per_audio_token (float, optional): number of frames corresponding to one audio token in the outputted
            audio. Defaults to 1024*24000/22050.
        alignment_method (str, optional): The method used to extract alignments.
            Defaults to "mean_per_word".
        strict_increase (bool, optional): Set to True to force alignments to be strictly increasing (once a token
            corresponds to the next word, it cannot be associated with a previous word).

    Returns:
        tuple: words durations in number of frames, in seconds,
            and the corresponding alignments between audio tokens and text tokens.
    """

    num_audio_tokens = len(attentions)  # i
    num_layers = len(attentions[0])  # 30
    num_heads = attentions[0][0].shape[1]  # 16
    alignments = []
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
        if alignment_method == "argmax_word_id_text_token":
            alignments.append(
                np.argmax([np.max(avg_scores[words_ids[i][0] : words_ids[i][1]]) for i in range(len(words_ids))])
            )
        # take id of word that has the max mean over text tokens
        elif alignment_method == "mean_per_word":
            alignments.append(
                np.argmax([np.mean(avg_scores[words_ids[i][0] : words_ids[i][1]]) for i in range(len(words_ids))])
            )

    if strict_increase:
        # Replace alignments to have only increasing words_ids in alignments
        for i in range(len(alignments) - 1):
            if alignments[i + 1] < alignments[i]:
                alignments[i + 1] = alignments[i]

    # Count words durations in number of associated audio tokens
    words_durations_in_nb_audio_token = np.zeros(len(words_ids))
    for idx in alignments:
        words_durations_in_nb_audio_token[idx] += 1

    # Count words durations in nb associated audio frames (converting audio token to frames)
    # nb_frames = nb_audio_token * gpt_code_stride_len * output_framerate * input_framerate = nb_audio_token * 1024 * 24000 / 22050 # upscaling
    nb_frames_per_audio_token = gpt_code_stride_len * output_sample_rate / internal_sample_rate
    words_durations_in_nb_frames = words_durations_in_nb_audio_token * nb_frames_per_audio_token

    # Convert token counts to time durations (in seconds)
    # time_per_token = 1024 / 22050  # 0.04644 seconds
    time_per_token = gpt_code_stride_len / internal_sample_rate  # 1024 / 22050 = 0.04644 seconds
    words_durations_in_sec = words_durations_in_nb_audio_token * time_per_token

    return (
        words_durations_in_nb_frames,
        words_durations_in_sec,
        alignments,
    )


def split_text_tokens_into_words(
    text_tokens,
    detokenized_text_tokens=[],
    space_token=2,
    split_method="space_remove_punctuation_only_words",
    rm_lang=True,
):
    """
    Splits a list of text tokens and its corresponding list of detokenized text into words based on space tokens and
    taking into account punctuation. The list of starting and ending ids of each word in the text token list is returned

    Args:
        text_tokens (torch.Tensor): The text tokens tensor.
        detokenized_text_tokens (list, optional): List of detokenized text. Defaults to [].
        space_token (int, optional): The token ID representing a space. Defaults to 2.
        split_method (str, optional): The method used to split into words.
            Defaults to "space_remove_punctuation_only_words".
        rm_lang (bool, optional): Set to False to keep the token corresponding to the language in the words_ids.
            Defaults to True.

    Returns:
        list: list of starting and ending ids of each word in the text token list.
    """
    punctuation_list = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "..."]
    space_token_ids = np.where(text_tokens.cpu() == space_token)[0]
    # punctuation_token_ids = [i for i, text in enumerate(detokenized_text_tokens) if text in punctuation_list]
    if split_method == "space":
        all_indices = [0] + space_token_ids.tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i], all_indices[i + 1]] for i in range(len(all_indices) - 1)]
    elif split_method == "space_end":
        all_indices = [0] + (space_token_ids + 1).tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i], all_indices[i + 1]] for i in range(len(all_indices) - 1)]
    elif split_method == "exclude_space":
        all_indices = [-1] + space_token_ids.tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i] + 1, all_indices[i + 1]] for i in range(len(all_indices) - 1)]
    elif split_method == "space_remove_punctuation_only_words":
        all_indices = [0] + space_token_ids.tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i], all_indices[i + 1]] for i in range(len(all_indices) - 1)]
        ids_to_pop = []
        for i in range(len(words_ids)):
            s_w, e_w = words_ids[i][0], words_ids[i][1]
            if e_w - s_w <= 2:
                if detokenized_text_tokens[e_w - 1] in punctuation_list:
                    ids_to_pop.append(i)
        for i in ids_to_pop[::-1]:
            words_ids.pop(i)
    elif split_method == "space_end_remove_punctuation_only_words":
        all_indices = [0] + (space_token_ids + 1).tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i], all_indices[i + 1]] for i in range(len(all_indices) - 1)]
        ids_to_pop = []
        for i in range(len(words_ids)):
            s_w, e_w = words_ids[i][0], words_ids[i][1]
            if e_w - s_w <= 2:
                if detokenized_text_tokens[s_w] in punctuation_list:
                    ids_to_pop.append(i)
        for i in ids_to_pop[::-1]:
            words_ids.pop(i)
    elif split_method == "exclude_space_remove_punctuation_only_words":
        all_indices = [-1] + space_token_ids.tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i] + 1, all_indices[i + 1]] for i in range(len(all_indices) - 1)]
        ids_to_pop = []
        for i in range(len(words_ids)):
            s_w, e_w = words_ids[i][0], words_ids[i][1]
            if e_w - s_w == 1:
                if detokenized_text_tokens[s_w] in punctuation_list:
                    ids_to_pop.append(i)
        for i in ids_to_pop[::-1]:
            words_ids.pop(i)
    elif split_method == "space_remove_punctuation":
        all_indices = [0] + space_token_ids.tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i], all_indices[i + 1]] for i in range(len(all_indices) - 1)]
        ids_to_pop = []
        for i in range(len(words_ids)):
            s_w, e_w = words_ids[i][0], words_ids[i][1]
            if e_w - s_w <= 2:
                if detokenized_text_tokens[e_w - 1] in punctuation_list:
                    ids_to_pop.append(i)
            elif detokenized_text_tokens[e_w - 1] in punctuation_list:
                words_ids[i][1] -= 1  # remove the first token (punctuation token)
        for i in ids_to_pop[::-1]:
            words_ids.pop(i)
    elif split_method == "space_end_remove_punctuation":
        all_indices = [0] + (space_token_ids + 1).tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i], all_indices[i + 1]] for i in range(len(all_indices) - 1)]
        ids_to_pop = []
        for i in range(len(words_ids)):
            s_w, e_w = words_ids[i][0], words_ids[i][1]
            if e_w - s_w <= 2:
                if detokenized_text_tokens[s_w] in punctuation_list:
                    ids_to_pop.append(i)
            elif detokenized_text_tokens[s_w] in punctuation_list:
                words_ids[i][0] += 1  # remove the first token (punctuation token)
        for i in ids_to_pop[::-1]:
            words_ids.pop(i)
    elif split_method == "exclude_space_remove_punctuation":
        all_indices = [-1] + space_token_ids.tolist() + [len(text_tokens)]
        words_ids = [[all_indices[i] + 1, all_indices[i + 1]] for i in range(len(all_indices) - 1)]
        ids_to_pop = []
        for i in range(len(words_ids)):
            s_w, e_w = words_ids[i][0], words_ids[i][1]
            if e_w - s_w == 1:
                if detokenized_text_tokens[s_w] in punctuation_list:
                    ids_to_pop.append(i)
            elif detokenized_text_tokens[s_w] in punctuation_list:
                words_ids[i][0] += 1  # remove the first token (punctuation token)
            elif detokenized_text_tokens[e_w - 1] in punctuation_list:
                words_ids[i][1] -= 1  # remove the first token (punctuation token)
        for i in ids_to_pop[::-1]:
            words_ids.pop(i)
    if rm_lang:
        words_ids[0][0] += 1  # remove the first token (language token)
    return words_ids


def get_words_durations_from_xtts_output(alignment_required_data):
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
    detokenized_text_tokens = alignment_required_data["detokenized_text_tokens"]
    internal_sample_rate = alignment_required_data["internal_sample_rate"]
    output_sample_rate = alignment_required_data["output_sample_rate"]
    gpt_code_stride_len = alignment_required_data["gpt_code_stride_len"]

    words_durations_in_nb_frames_list = []
    words_durations_in_sec_list = []
    alignments_list = []
    if len(att_attentions) > 1:
        print("more than one sentence, returning empty lists")
    for i in range(len(att_attentions)):

        words_ids = split_text_tokens_into_words(
            text_tokens=text_tokens[i], detokenized_text_tokens=detokenized_text_tokens[i]
        )
        # k_len = gpt_cond_latent_shape[1] + bos + num_text_tokens + eos + start_audio_token
        start_text_token = gpt_cond_latent_shape[i][1] + 1
        end_text_token = gpt_cond_latent_shape[i][1] + 1 + len(text_tokens[i])
        words_durations_in_nb_frames, words_durations_in_sec, alignments = extract_words_durations(
            attentions=att_attentions[i],
            num_text_tokens=len(text_tokens[i]),
            start_text_token=start_text_token,
            end_text_token=end_text_token,
            words_ids=words_ids,
            internal_sample_rate=internal_sample_rate,
            output_sample_rate=output_sample_rate,
            gpt_code_stride_len=gpt_code_stride_len,
            # sample_rate=24000,
            # alignment_method="mean_per_word",  # mean_per_word, argmax_word_id_text_token, argmax_text_token
            # strict_increase=True,
        )
        words_durations_in_nb_frames_list.extend(words_durations_in_nb_frames)
        words_durations_in_sec_list.extend(words_durations_in_sec)
        alignments_list.extend(alignments)
    return words_durations_in_nb_frames_list, words_durations_in_sec_list, alignments_list


def test_different_alignment_and_wordsplit(alignment_required_data):
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
    detokenized_text = alignment_required_data["detokenized_text"]
    internal_sample_rate = alignment_required_data["internal_sample_rate"]
    output_sample_rate = alignment_required_data["output_sample_rate"]
    gpt_code_stride_len = alignment_required_data["gpt_code_stride_len"]

    if len(att_attentions) > 1:
        print("more than one sentence, returning empty lists")
    else:
        att_attentions = att_attentions[0]
        gpt_cond_latent_shape = gpt_cond_latent_shape[0]
        text_tokens = text_tokens[0]
        detokenized_text_tokens = detokenized_text_tokens[0]
        detokenized_text = detokenized_text[0]

        words_durations_seconds_dict = {}
        start_text_token = gpt_cond_latent_shape[1] + 1
        end_text_token = gpt_cond_latent_shape[1] + 1 + len(text_tokens)
        for word_splitting_method in [
            "space",
            "space_end",
            "exclude_space",
            "space_remove_punctuation_only_words",
            "space_remove_punctuation",
            "space_end_remove_punctuation_only_words",
            "space_end_remove_punctuation",
            "exclude_space_remove_punctuation_only_words",
            "exclude_space_remove_punctuation",
        ]:  # "nltk_word_tokenize",
            words_ids = split_text_tokens_into_words(
                text_tokens=text_tokens,
                detokenized_text_tokens=detokenized_text_tokens,
                split_method=word_splitting_method,
            )
            if word_splitting_method not in words_durations_seconds_dict:
                words_durations_seconds_dict[word_splitting_method] = {
                    "words_ids": words_ids,
                    "splitted_words": ["".join(detokenized_text_tokens[w[0] : w[1]]) for w in words_ids],
                    "alignments": {},
                }
            for alignment_method in [
                "mean_per_word",
                "argmax_word_id_text_token",
            ]:
                words_durations_in_nb_frames, words_durations_in_sec, alignments = extract_words_durations(
                    attentions=att_attentions,
                    num_text_tokens=len(text_tokens),
                    start_text_token=start_text_token,
                    end_text_token=end_text_token,
                    words_ids=words_ids,
                    internal_sample_rate=internal_sample_rate,
                    output_sample_rate=output_sample_rate,
                    gpt_code_stride_len=gpt_code_stride_len,
                    # sample_rate=24000,
                    # alignment_method="mean_per_word",  # mean_per_word, argmax_word_id_text_token, argmax_text_token
                    # strict_increase=True,
                )
                words_durations_seconds_dict[word_splitting_method]["alignments"][
                    alignment_method
                ] = words_durations_in_sec.tolist()
    return words_durations_seconds_dict
