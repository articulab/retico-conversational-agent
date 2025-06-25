import argparse
from datetime import datetime
from functools import partial
import pandas as pd
import torch
import retico_conversational_agent as agent
from os import walk
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":

    tts_latency_folder = "tts_latency/"

    all_tts_latencies_files = [f for f in listdir(tts_latency_folder) if isfile(join(tts_latency_folder, f))]
    print(all_tts_latencies_files)

    for file in all_tts_latencies_files:
        path = tts_latency_folder + file
        df = pd.read_csv(path)
