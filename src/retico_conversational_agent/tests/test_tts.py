import os
import re
import wave
from TTS.api import TTS
import retico_core
import soundfile as sf
import torch

#### COMMAND
# tts --text "To have much learning and skill, to be well-trained in discipline, and good in speech — this is the highest blessing." --language_idx "en" --out_path "audio_tests/xtts.wav" --model_name "tts_models/multilingual/multi-dataset/xtts_v2"  --speaker_idx "Claribel Dervla"


def append_raw_audio(raw_audio, rate, dir_path, file_path):
    if isinstance(raw_audio, str):
        audio_data = eval(raw_audio)
    else:
        audio_data = raw_audio
    os.makedirs(dir_path, exist_ok=True)
    with sf.SoundFile(
        file_path,
        mode="x+" if not os.path.exists(file_path) else "r+",
        samplerate=rate if not os.path.exists(file_path) else None,
        channels=1 if not os.path.exists(file_path) else None,
    ) as f:
        wav_data = retico_core.audio.convert_audio_PCM16_to_float32(raw_audio=audio_data)
        f.seek(0, sf.SEEK_END)  # Append to end of file
        f.write(wav_data)


def generate_voices_speakers(speaker_ids, model, text):
    for speaker_id in speaker_ids:
        file_path = f"audio_tests/xtts/{speaker_id.replace(' ', '_')}.wav"
        model.tts_to_file(
            text=text,
            file_path=file_path,
            speaker=speaker_id,
            language="en",
        )


def concatenate_audio_wave(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using Python's built-in wav module
    and save it to `output_path`. Note that extension (wav) must be added to `output_path`"""
    data = []
    for clip in audio_clip_paths:
        w = wave.open(clip, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    output = wave.open(output_path, "wb")
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()


def generate_audio_splitted_sentences(model, speaker_id, sentences, punctuation_str):
    speaker_path = speaker_id.replace(" ", "_")
    for i, sentence in enumerate(sentences):
        # generate audio for the full sentence
        file_path = f"audio_tests/xtts/{speaker_path}/splitted_sentences/sentence_{i}/full_sentence.wav"
        folder_path = "/".join(file_path.split("/")[:-1])
        os.makedirs(folder_path, exist_ok=True)
        model.tts_to_file(
            text=sentence,
            file_path=file_path,
            speaker=speaker_id,
            language="en",
        )

        clauses = re.split(punctuation_str, sentence)
        clauses = [c.lstrip() for c in clauses if c != ""]
        audio_bytes = b""
        # generate audio for the clauses
        clauses_paths = []
        for j, clause in enumerate(clauses):
            file_path = f"audio_tests/xtts/{speaker_path}/splitted_sentences/sentence_{i}/clause_{j}.wav"
            folder_path = "/".join(file_path.split("/")[:-1])
            os.makedirs(folder_path, exist_ok=True)
            model.tts_to_file(
                text=clause,
                file_path=file_path,
                speaker=speaker_id,
                language="en",
            )
            clauses_paths.append(file_path)
            # audio, _ = retico_core.audio.load_audiofile_WAVPCM16(file_path)
            # audio, _ = retico_core.audio.load_audiofile_PCM16(file_path)
            # audio_bytes += audio

        # aggregate all clauses audio files into one
        file_path = f"audio_tests/xtts/{speaker_path}/splitted_sentences/sentence_{i}/aggregated_clauses.wav"
        os.makedirs(folder_path, exist_ok=True)
        concatenate_audio_wave(clauses_paths, file_path)
        # with open(file_path, "wb") as f:
        #     f.write(audio_bytes)


speaker_ids = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Andrew Chipper",
    "Badr Odhiambo",
    "Dionisio Schuyler",
    "Royston Min",
    "Viktor Eka",
    "Abrahan Mack",
    "Adde Michal",
    "Baldur Sanjin",
    "Craig Gutsy",
    "Damien Black",
    "Gilberto Mathias",
    "Ilkin Urbano",
    "Kazuhiko Atallah",
    "Ludvig Milivoj",
    "Suad Qasim",
    "Torcull Diarmuid",
    "Viktor Menelaos",
    "Zacharie Aimilios",
    "Nova Hogarth",
    "Maja Ruoho",
    "Uta Obando",
    "Lidiya Szekeres",
    "Chandra MacFarland",
    "Szofi Granger",
    "Camilla Holmström",
    "Lilya Stainthorpe",
    "Zofija Kendrick",
    "Narelle Moon",
    "Barbora MacLean",
    "Alexandra Hisakawa",
    "Alma María",
    "Rosemary Okafor",
    "Ige Behringer",
    "Filip Traverse",
    "Damjan Chapman",
    "Wulf Carlevaro",
    "Aaron Dreschner",
    "Kumar Dahl",
    "Eugenio Mataracı",
    "Ferran Simen",
    "Xavier Hayasaka",
    "Luis Moray",
    "Marcos Rudaski",
]
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
# model_name = "tts_models/en/jenny/jenny"
language = "en"
device = "cuda" if torch.cuda.is_available() else "cpu"
frame_duration = 0.2
samplewidth = 2
verbose = True
text = "To have much learning and skill, to be well-trained in discipline, and good in speech — this is the highest blessing."
folder_path = "audio_tests/xtts/"
model = TTS(model_name).to(device)
samplerate = model.synthesizer.tts_config.get("audio")["sample_rate"]
file_path = f"audio_tests/{model_name.split('/')[-1]}.wav"


# speaker_id = "Gitta Nikolina"
# speaker_id = "Daisy Studious"
# speaker_id = "Alma María"
speaker_id = "Uta Obando"


sentences = [
    "To have much learning and skill, to be well-trained in discipline, and good in speech — this is the highest blessing.",
    "A passion for politics stems usually from an insatiable need, either for power, or for friendship and adulation, or a combination of both.",
    "Change will not come if we wait for some other person or some other time. We are the ones we've been waiting for. We are the change that we seek.",
]
punctuation = [".", ",", ";", ":", "!", "?"]
punctuation_str = r"[.,;!?]+"

generate_audio_splitted_sentences(
    model=model, speaker_id=speaker_id, sentences=sentences, punctuation_str=punctuation_str
)
# new_audio, final_outputs = synthesize(current_text)
# final_outputs = model.tts(
#     text=text,
#     return_extra_outputs=True,
#     split_sentences=False,
#     verbose=verbose,
# )
# final_outputs = model.tts(
#     text=text,
#     language=language,
#     speaker="p225",
#     # speaker_wav=speaker_wav,
#     # speaker="Ana Florence",
# )
# waveforms, outputs = final_outputs
# waveform = retico_core.audio.convert_audio_float32_to_PCM16(raw_audio=waveforms)

# append_raw_audio(
#     raw_audio=waveform,
#     rate=samplerate,
#     dir_path=folder_path,
#     file_path=file_path,
# )
