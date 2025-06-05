import json
import os
import wave

from faster_whisper import WhisperModel
import torch
from TTS.api import TTS
import numpy as np
import retico_core
import alignment_xtts


def synthesize(text, is_multilingual, model, language, speaker_id=None, speaker_wav=None):
    """Takes the given text and synthesizes speech using the TTS model.
    Returns the synthesized speech as 22050 Hz int16-encoded numpy ndarray.

    Args:
        text (str): The text to use to synthesize speech.

    Returns:
        bytes: The speech as a 22050 Hz int16-encoded numpy ndarray.
    """

    if is_multilingual:
        final_outputs = model.tts(
            text=text,
            language=language,
            speaker=speaker_id,
            speaker_wav=None if speaker_id is None else speaker_wav,
            return_extra_outputs=True,
            split_sentences=False,
            verbose=True,
        )
    else:
        final_outputs = model.tts(
            text=text,
            return_extra_outputs=True,
            split_sentences=False,
            verbose=True,
        )

    if len(final_outputs) != 2:
        raise NotImplementedError("coqui TTS should output both wavforms and outputs")
    else:
        waveforms, outputs = final_outputs

    return outputs

    # waveform = retico_core.audio.convert_audio_float32_to_PCM16(raw_audio=waveforms)

    # return waveform, outputs


def recognize(model, audio_data, samplerate=24000, whisper_samplerate=16000):
    """Recreate the audio signal received by the microphone by
    concatenating the audio chunks from the audio_buffer and transcribe
    this concatenation into a list of predicted words.

    Returns:
        (list[string], boolean): the list of transcribed words.
    """
    transcript = model.transcribe(audio_data, word_timestamps=True)
    segments, infos = transcript
    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                print(
                    f"  Word: '{word.word}' at {word.start*whisper_samplerate/samplerate:.2f}s - {word.end*whisper_samplerate/samplerate:.2f}s"
                )
                words.append(
                    [
                        word.word,
                        word.start * whisper_samplerate / samplerate,
                        word.end * whisper_samplerate / samplerate,
                    ]
                )

    return words


def compute_mae_rmse(predicted_starts, ground_truth_starts):
    """
    Compute MAE and RMSE between predicted and ground-truth start times.

    Args:
        predicted_starts (list or np.array): List of predicted start times (in seconds).
        ground_truth_starts (list or np.array): List of ground-truth start times (in seconds).

    Returns:
        tuple: (mae, rmse)
    """
    predicted_starts = np.array(predicted_starts)
    ground_truth_starts = np.array(ground_truth_starts)

    # Ensure the same length
    assert len(predicted_starts) == len(ground_truth_starts), "Lists must be of equal length."

    errors = predicted_starts - ground_truth_starts
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    return mae, rmse


def evaluate_alignments(
    text, model, is_multilingual, language, speaker_id, asr_model="distil-large-v2", device="cuda", compute_type="int8"
):

    asr = WhisperModel(asr_model, device=device, compute_type=compute_type)

    current_text = text
    words = current_text.split(" ")
    outputs = synthesize(current_text, is_multilingual, model, language, speaker_id=speaker_id)

    assert len(outputs) == 1  # only one clause, one sentence

    if is_multilingual:
        tokens = model.synthesizer.tts_model.tokenizer.encode(current_text, lang=language + "-")
        space_token = model.synthesizer.tts_model.tokenizer.encode(" ", lang=language + "-")[0]
    else:
        tokens = model.synthesizer.tts_model.tokenizer.text_to_ids(current_text)
        space_token = model.synthesizer.tts_model.tokenizer.encode(" ")[0]

    audio_words_ends = []
    for i, x in enumerate(tokens):
        if x == space_token or i == len(tokens) - 1:
            audio_words_ends.append(i + 1)
    audio_data = outputs[0]["wav"]
    alignment_required_data = outputs[0]["alignment_required_data"]

    words_durations_seconds_dict = alignment_xtts.test_different_alignment_and_wordsplit(alignment_required_data)
    for word_splitting_method in words_durations_seconds_dict.keys():
        print(
            f" splitted words {word_splitting_method} \n{words_durations_seconds_dict[word_splitting_method]['splitted_words']}"
        )
    with open("words_durations.json", "w") as f:
        json.dump(words_durations_seconds_dict, f, ensure_ascii=False, indent=4)

    asr_timestamps = recognize(asr, audio_data)
    asr_end_timestamps = [w[2] for w in asr_timestamps]
    with open("words_durations_asr.json", "w") as f:
        json.dump(asr_end_timestamps, f, ensure_ascii=False, indent=4)

    words_timestamps_extraction_similarity_scores = {}
    for word_splitting_method in words_durations_seconds_dict.keys():
        if word_splitting_method not in words_timestamps_extraction_similarity_scores:
            words_timestamps_extraction_similarity_scores[word_splitting_method] = {}
        for alignment_method in words_durations_seconds_dict[word_splitting_method]["alignments"].keys():
            words_durations = words_durations_seconds_dict[word_splitting_method]["alignments"][alignment_method]
            words_end_timestamps = np.cumsum(words_durations)

            if len(words_end_timestamps) >= len(asr_end_timestamps):
                words_end_timestamps = words_end_timestamps[: len(asr_end_timestamps)]
            elif len(words_end_timestamps) < len(asr_end_timestamps):
                asr_end_timestamps = asr_end_timestamps[: len(words_end_timestamps)]

            mae, rmse = compute_mae_rmse(
                words_end_timestamps,
                asr_end_timestamps,
            )
            print(f"MAE: {mae:.3f} seconds")
            print(f"RMSE: {rmse:.3f} seconds")
            words_timestamps_extraction_similarity_scores[word_splitting_method][alignment_method] = {
                "mae": mae,
                "rmse": rmse,
            }

    with open("words_timestamps_extraction_similarity_scores.json", "w") as f:
        json.dump(words_timestamps_extraction_similarity_scores, f, ensure_ascii=False, indent=4)


def evaluate_alignments_files(
    word_durations_file="words_durations.json",
    asr_file="words_durations_asr.json",
    res_file="words_timestamps_extraction_similarity_scores.json",
):
    with open(word_durations_file, "r") as f:
        words_durations_seconds_dict = json.load(f)

    print("words_durations_seconds_dict", words_durations_seconds_dict)

    with open(asr_file, "r") as f:
        asr_end_timestamps = json.load(f)

    print("asr_end_timestamps", asr_end_timestamps)

    words_timestamps_extraction_similarity_scores = {}
    for word_splitting_method in words_durations_seconds_dict.keys():
        if word_splitting_method not in words_timestamps_extraction_similarity_scores:
            words_timestamps_extraction_similarity_scores[word_splitting_method] = {}
        for alignment_method in words_durations_seconds_dict[word_splitting_method]["alignments"].keys():
            if alignment_method not in words_timestamps_extraction_similarity_scores[word_splitting_method]:
                words_timestamps_extraction_similarity_scores[word_splitting_method][alignment_method] = {}
            words_durations = words_durations_seconds_dict[word_splitting_method]["alignments"][alignment_method]
            words_end_timestamps = np.cumsum(words_durations)

            print("words_end_timestamps", words_end_timestamps)

            mae, rmse = compute_mae_rmse(
                words_end_timestamps,
                asr_end_timestamps,
            )
            print(f"MAE: {mae:.3f} seconds")
            print(f"RMSE: {rmse:.3f} seconds")
            words_timestamps_extraction_similarity_scores[word_splitting_method][alignment_method] = {
                "mae": mae,
                "rmse": rmse,
            }

    with open(res_file, "w") as f:
        json.dump(words_timestamps_extraction_similarity_scores, f, ensure_ascii=False, indent=4)


def execute_tts(text, model, is_multilingual, language, speaker_id, samplerate, asr, folder_save_audio):
    current_text = text
    words = current_text.split(" ")
    outputs = synthesize(current_text, is_multilingual, model, language, speaker_id=speaker_id)

    assert len(outputs) == 1  # only one clause, one sentence

    if is_multilingual:
        tokens = model.synthesizer.tts_model.tokenizer.encode(current_text, lang=language + "-")
        space_token = model.synthesizer.tts_model.tokenizer.encode(" ", lang=language + "-")[0]
        # tokens = self.model.synthesizer.tts_model.tokenizer.tokenizer.text_to_ids(current_text)
    else:
        tokens = model.synthesizer.tts_model.tokenizer.text_to_ids(current_text)
        space_token = model.synthesizer.tts_model.tokenizer.encode(" ")[0]

    audio_words_ends = []
    for i, x in enumerate(tokens):
        if x == space_token or i == len(tokens) - 1:
            audio_words_ends.append(i + 1)
    audio_data = outputs[0]["wav"]

    alignment_required_data = outputs[0]["alignment_required_data"]

    words_durations_in_nb_frames, words_durations_in_sec, alignments = (
        alignment_xtts.get_words_durations_from_xtts_output(alignment_required_data)
    )

    # print("words_durations_in_nb_frames", words_durations_in_nb_frames)
    words_end_timestamps = np.cumsum(words_durations_in_sec)
    print("words_timestamps", np.round(words_end_timestamps, 2))

    asr_timestamps = recognize(asr, audio_data)
    print("asr_timestamps", asr_timestamps)
    assert len(asr_timestamps) == len(words_durations_in_sec)

    # save complete audio
    os.makedirs(folder_save_audio, exist_ok=True)
    sample_width = 2
    wav_data_complete = retico_core.audio.convert_audio_float32_to_WAVPCM16(raw_audio=audio_data, samplerate=samplerate)
    with wave.open(f"{folder_save_audio}/complete_audio.wav", "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(sample_width)  # Sample width in bytes
        wav_file.setframerate(samplerate)  # Sample rate
        wav_file.writeframes(wav_data_complete)

    # save chunks with internal splitting
    previous_duration = 0
    os.makedirs(f"{folder_save_audio}/internal_split/", exist_ok=True)
    for i, duration_len in enumerate(words_durations_in_nb_frames):
        start = int(previous_duration)
        end = int(previous_duration + duration_len)
        # print("start", start, "end", end)
        audio_chunk = audio_data[start:end]
        audio_chunk_wav = retico_core.audio.convert_audio_float32_to_WAVPCM16(
            raw_audio=audio_chunk, samplerate=samplerate
        )
        with wave.open(f"{folder_save_audio}/internal_split/audio_chunk_{i}.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)  # Sample width in bytes
            wav_file.setframerate(samplerate)  # Sample rate
            wav_file.writeframes(audio_chunk_wav)
        previous_duration += duration_len

    # save chunks with whisper splitting
    previous_duration = 0
    os.makedirs(f"{folder_save_audio}/whisper_split/", exist_ok=True)
    for i, w in enumerate(asr_timestamps):
        word, start_s, end_s = w
        start_frames = int(start_s * samplerate)
        end_frames = int(end_s * samplerate)
        audio_chunk = audio_data[start_frames:end_frames]
        audio_chunk_wav = retico_core.audio.convert_audio_float32_to_WAVPCM16(
            raw_audio=audio_chunk, samplerate=samplerate
        )
        with wave.open(f"{folder_save_audio}/whisper_split/audio_chunk_{i}.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)  # Sample width in bytes
            wav_file.setframerate(samplerate)  # Sample rate
            wav_file.writeframes(audio_chunk_wav)
        previous_duration += duration_len

    words_last_frame = np.cumsum(words_durations_in_nb_frames).tolist()
    print("current_text", current_text)
    print("words", words)
    chunk_size = 512
    chunk_size_bytes = chunk_size * sample_width
    # Split the audio into same-size chunks
    for chunk_start in range(0, len(audio_data), chunk_size):
        chunk_wav = audio_data[chunk_start : chunk_start + chunk_size]
        chunk = (np.array(chunk_wav) * 32767).astype(np.int16).tobytes()
        # chunk = retico_core.audio.convert_audio_float32_to_PCM16(raw_audio=chunk)
        if len(chunk) < chunk_size_bytes:
            chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
        # Calculates last word that started during the audio chunk
        word_id = len([1 for word_end in words_last_frame if word_end < chunk_start + chunk_size])

        word_id = min(word_id, len(words) - 1)
        # grounded_iu = clause_ius[word_id]
        grounded_word = words[word_id]
        char_id = sum([len(word) for word in words[: word_id + 1]]) - 1
        print("word_id", word_id, "char_id", char_id)
        print("grounded_word", grounded_word, "sentence_stop", current_text[: char_id + 1])
        # iu = self.create_iu(
        #     grounded_in=grounded_iu,
        #     raw_audio=chunk,
        #     chunk_size=chunk_size,
        #     rate=samplerate,
        #     sample_width=sample_width,
        #     grounded_word=words[word_id],
        #     word_id=int(word_id),
        #     char_id=char_id,
        #     turn_id=grounded_iu.turn_id,
        #     clause_id=grounded_iu.clause_id,
        # )
        # new_buffer.append(iu)
        # return new_buffer


if __name__ == "__main__":

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    # model_name = "tts_models/en/jenny/jenny"
    speaker_id = "Gitta Nikolina"
    language = "en"
    is_multilingual = "multilingual" in model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TTS(model_name).to(device)
    samplerate = 24000
    folder_save_audio = "audios_alignment/"
    asr_model = "distil-large-v2"

    # text = "Hello, how are you?"
    # text = "How are you doing this sunny morning? Are you good?"
    text = "Hello. How are you doing this sunny morning, Are you good ? is it -like the other day- a sunny day? It's looking like it !"
    # text = "Change will not come if we wait for some other person or some other time."
    # text = "Yes. Now you know if everybody like in August when everybody's on vacation, or something ... we can dress a little more casual or?"

    asr = WhisperModel("distil-large-v2", device=device, compute_type="int8")

    execute_tts(
        text=text,
        model=model,
        is_multilingual=is_multilingual,
        language=language,
        speaker_id=speaker_id,
        samplerate=samplerate,
        asr=asr,
        folder_save_audio=folder_save_audio,
    )

    evaluate_alignments(
        text=text,
        model=model,
        is_multilingual=is_multilingual,
        language=language,
        speaker_id=speaker_id,
        asr_model=asr_model,
    )
