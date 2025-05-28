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


def execute_tts(text, model, is_multilingual, language, speaker_id, samplerate, asr):
    current_text = text
    words = current_text.split(" ")
    outputs = synthesize(current_text, is_multilingual, model, language, speaker_id=speaker_id)

    assert len(outputs) == 1  # only one clause, one sentence
    print("outputs keys = ", outputs[0].keys())

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
    # words_durations_in_sec = outputs[0]["outputs"]["words_durations_in_sec"].squeeze().tolist()
    # words_durations_in_nb_frames = outputs[0]["outputs"]["words_durations_in_nb_frames"].squeeze().tolist()

    words_durations_in_nb_frames, words_durations_in_sec, alignments = alignment_xtts.get_words_durations(
        outputs[0]["alignment_required_data"]
    )

    # print("words_durations_in_nb_frames", words_durations_in_nb_frames)
    print("words_timestamps", np.round(np.cumsum(words_durations_in_sec), 2))

    asr_timestamps = recognize(asr, audio_data)
    assert len(asr_timestamps) == len(words_durations_in_sec)

    # assert (words_durations_in_nb_frames2 == words_durations_in_nb_frames).all()

    # if isinstance(raw_audio, str):
    #     audio_data = eval(raw_audio)
    # else:
    #     audio_data = raw_audio

    # print("len audio_data", len(audio_data))
    # print("samplerate", samplerate)
    # print("duration audio", len(audio_data) / samplerate)
    # print("duration audio", len(audio_data) / 22050)
    # print("duration audio", len(audio_data) / 16000)
    # with open("audios_tts/audio.wav", "wb") as f:
    #     f.write(wav_data)

    os.makedirs("audios_tts/", exist_ok=True)
    sample_width = 2
    wav_data_complete = retico_core.audio.convert_audio_float32_to_WAVPCM16(raw_audio=audio_data, samplerate=samplerate)
    with wave.open("audios_tts/audio.wav", "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(sample_width)  # Sample width in bytes
        wav_file.setframerate(samplerate)  # Sample rate
        wav_file.writeframes(wav_data_complete)

    # audio chunks
    previous_duration = 0
    # for i, words_duration in enumerate(words_durations):
    #     start = int(previous_duration * samplerate)
    #     end = int(previous_duration + words_duration * samplerate)
    #     print("start", start, "end", end)
    #     audio_chunk = wav_data[start:end]
    #     with open(f"audios_tts/audio_chunk_{i}.wav", "wb") as f:
    #         f.write(audio_chunk)
    #     previous_duration += words_duration
    for i, duration_len in enumerate(words_durations_in_nb_frames):
        start = int(previous_duration)
        end = int(previous_duration + duration_len)
        # print("start", start, "end", end)
        audio_chunk = audio_data[start:end]
        audio_chunk_wav = retico_core.audio.convert_audio_float32_to_WAVPCM16(
            raw_audio=audio_chunk, samplerate=samplerate
        )
        with wave.open(f"audios_tts/audio_chunk_{i}.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)  # Sample width in bytes
            wav_file.setframerate(samplerate)  # Sample rate
            wav_file.writeframes(audio_chunk_wav)
        previous_duration += duration_len

    # durations = outputs[0]["outputs"]["durations"].squeeze().tolist()
    # len_wav = len(audio_data)
    # total_duration = int(sum(durations))
    # nb_fram_per_dur = len_wav / total_duration
    # new_buffer = []

    # Check that input words matches synthesized audio
    # att = outputs[0]["outputs"]["att"]
    # tokens_gpt = outputs[0]["outputs"]["gpt_codes"].squeeze().tolist()
    # att_align = outputs[0]["outputs"]["att_alignment"]
    # token_timestamps = outputs[0]["outputs"]["token_timestamps"]
    # print(
    #     "att shape",
    #     len(att),
    #     len(att[0]),
    #     att[0][0].shape,
    #     [len(i) for i in att],
    # )
    # nb text tokens = 14
    # nb gpt tokens (after forward in gpt) = 25

    # nb text tokens = 31
    # nb gpt tokens (after forward in gpt) = 71

    # att shape is nb_gpt_tokens x 30 (layers ?) x [1, 16, 49, 49] or [1, 16, 1, 49+i]for i in range(1, nb_gpt_tokens)
    # att shape is nb_gpt_tokens x 30 x 16 x c+i for i in range(1, nb_gpt_tokens)
    # -> what is 16, 30 ? head and layers ?
    # 49 is the number of frames in the audio
    # total_shape = [[j.shape for j in i] for i in att]
    # print("att shape 0", total_shape[0][0])
    # for idxi, i in enumerate(total_shape):
    #     for idxj, j in enumerate(i):
    #         if j != total_shape[0][0]:
    #             print(f"att shape {idxi}, {idxj}", j)
    # print("att_align shape", att_align.shape)
    # print("len token_timestamps", len(token_timestamps))
    # print("token_timestamps", token_timestamps)
    # print("len(audio_words_ends)", len(audio_words_ends))
    # print("len(words)", len(words))
    # print("len(durations)", len(durations))
    # print("len(tokens)", len(tokens))
    # print("len(tokens_gpt)", len(tokens_gpt))
    # print("tokens gpt", tokens_gpt)
    # print("tokens", tokens)
    # print("words", words)
    # print("self.space_token", space_token)
    # print("audio_words_ends", audio_words_ends)
    # assert len(audio_words_ends) == len(words)
    # assert len(durations) == len(tokens)

    # # calculate audio duration per word
    # words_duration = []
    # old_len_w = 0
    # for s_id in audio_words_ends:
    #     # words_duration.append(int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION)
    #     words_duration.append(int(sum(durations[old_len_w:s_id]) * nb_fram_per_dur))
    #     old_len_w = s_id
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
        # print("word_id", word_id, "char_id", char_id)
        # print("grounded_word", grounded_word, "sentence_stop", current_text[: char_id + 1])
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


model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
# model_name = "tts_models/en/jenny/jenny"
speaker_id = "Gitta Nikolina"
language = "en"
is_multilingual = "multilingual" in model_name
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TTS(model_name).to(device)
samplerate = 24000


# text = "Hello, how are you?"
# text = "How are you doing this sunny morning? Are you good?"
# text = "Hello. How are you doing this sunny morning, Are you good ? is it -like the other day- a sunny day? It's looking like it !"
text = "Change will not come if we wait for some other person or some other time."

asr = WhisperModel("distil-large-v2", device=device, compute_type="int8")

execute_tts(
    text=text,
    model=model,
    is_multilingual=is_multilingual,
    language=language,
    speaker_id=speaker_id,
    samplerate=samplerate,
    asr=asr,
)
