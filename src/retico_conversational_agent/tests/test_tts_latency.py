import argparse
from datetime import datetime
from functools import partial
import pandas as pd
import torch
import retico_conversational_agent as agent

if __name__ == "__main__":

    tts_models = {
        "xtts": partial(
            agent.TTS_DM_subclass.CoquiTTSSubclass,
            # model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            model_name="xtts",
        ),
        "jenny": partial(
            agent.TTS_DM_subclass.CoquiTTSSubclass,
            # model_name="tts_models/en/jenny/jenny",
            model_name="jenny",
        ),
        "glow": partial(
            agent.TTS_DM_subclass.CoquiTTSSubclass,
            model_name="glow",
        ),
        "your_tts": partial(
            agent.TTS_DM_subclass.CoquiTTSSubclass,
            model_name="your_tts",
        ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        help="The model to test.",
        type=str,
        choices=tts_models.keys(),
        required=True,
    )
    parser.add_argument(
        "--speaker-id",
        "-s",
        help="The speaker id to pick.",
        type=str,
        default="Gitta Nikolina",
    )
    parser.add_argument(
        "--speaker-wav",
        "-sw",
        help="The speaker wav to pick.",
        type=str,
        default="voices_different_tts/xtts/Uta_Obando/splitted_sentences/sentence_2/full_sentence.wav",
    )
    args = parser.parse_args()

    tts_model_name = args.model

    print("tts model", tts_model_name)

    # tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    # model_name = "tts_models/en/jenny/jenny"
    # tts_speaker_id = "Gitta Nikolina"
    language = "en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_latency_folder = "tts_latency/"
    verbose = False

    # tts = agent.TTS_DM_subclass.CoquiTTSSubclass(
    #     device=device,
    #     # incrementality_level=incrementality_level,
    #     # frame_duration=tts_frame_length,
    #     verbose=verbose,
    #     language="en",
    #     model_name=tts_model_name,
    #     speaker_id=tts_speaker_id,
    # )

    print("dict ", tts_models[tts_model_name])

    tts = tts_models[tts_model_name](
        device=device,
        verbose=verbose,
        language="en",
        speaker_id=args.speaker_id,
        speaker_wav=args.speaker_wav,
    )
    tts.setup()

    # text = "Hello, how are you?"
    # text = "How are you doing this sunny morning? Are you good?"
    text = "Hello. How are you doing this sunny morning, Are you good ? is it -like the other day- a sunny day? It's looking like it !"
    # text = "Change will not come if we wait for some other person or some other time."
    # text = "Yes. Now you know if everybody like in August when everybody's on vacation, or something ... we can dress a little more casual or?"
    texts = [text]
    texts = {
        "warming": [
            "Okay, so what do you think about us getting involved in the Middle East?",
            "Well, you don't find many intelligent people starting wars you know.",
            "I suppose that's probably true.",
        ],
        "sentences": [
            "Hello, how are you?",
            "How are you doing this sunny morning? Are you good?",
            "Hello. How are you doing this sunny morning, Are you good ? is it -like the other day- a sunny day? It's looking like it !",
            "Change will not come if we wait for some other person or some other time.",
            "Yes. Now you know if everybody like in August when everybody's on vacation, or something ... we can dress a little more casual or?",
        ],
        "clauses": [
            "Yes.",
            "No.",
            "Hey,",
            "Hello,",
            "Monday,",
            "Tuesday,",
            "Wednesday,",
            "Thursday,",
            "Friday,",
            "Which,",
            "So,",
            "Okay,",
            "Well,",
            "First,",
            "Then,",
            "how are you?",
            "How are you doing this sunny morning?",
            "Are you good?",
        ],
    }

    # punctuation_text = [".", ",", ";", ":", "!", "?"]
    # re.split('; |, |\*|\n', string_to_split)

    durations = []
    nb_words = []
    text_types = []
    for text_type in texts:
        print("text_type ", text_type)
        for text in texts[text_type]:
            start_time = datetime.now()
            tts.produce(current_text=text)
            end_time = datetime.now()
            duration = end_time - start_time
            duration_in_seconds = duration.total_seconds()
            print(f"duration = {duration_in_seconds}")
            durations.append(duration_in_seconds)
            words = text.split(" ")
            nb_words.append(len(words))
            text_types.append(text_type)

    dict_durations = {
        "duration": durations[len(texts["warming"]) :],
        "nb_words": nb_words[len(texts["warming"]) :],
        "text_type": text_types[len(texts["warming"]) :],
    }
    df = pd.DataFrame(dict_durations)
    file_path = tts_latency_folder + args.model + ".csv"
    df.to_csv(file_path)
