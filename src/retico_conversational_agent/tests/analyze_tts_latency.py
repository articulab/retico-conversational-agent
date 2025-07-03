# from datetime import datetime
# from functools import partial
# import torch
# import retico_conversational_agent as agent
# from os import walk
# import argparse
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


def plot_hist_width_consistent(dict_latencies, avg_duration_per_length):

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plot the first histogram

    width = 1 / (len(dict_latencies) + 1)
    for id_model, model in enumerate(dict_latencies):
        offset = id_model * width
        plt.bar(
            avg_duration_per_length[id_model].index - (1 - width) / 2 + offset,
            avg_duration_per_length[id_model].values,
            width=width,
            align="edge",
            label=model.split(".")[0],
        )

    # Adding labels and title
    plt.xlabel("Number of Words")
    plt.ylabel("Duration")
    plt.title("Histogram of Durations per Number of Words")
    plt.legend()

    # Show the plot
    plt.show()


def plot_hist_width_consistent_no_ids(dict_latencies, avg_duration_per_length):

    # Plotting the data
    fig, ax = plt.subplots(layout="constrained")
    # plt.figure(figsize=(10, 6))

    # Plot the first histogram
    x = np.arange(len(avg_duration_per_length[0]))
    width = 1 / (len(dict_latencies) + 1)
    for id_model, model in enumerate(dict_latencies):
        offset = id_model * width
        ax.bar(
            x - (1 - width) / 2 + offset,
            avg_duration_per_length[id_model].values,
            width=width,
            align="edge",
            label=model.split(".")[0],
        )
        # ax.bar_label(model.split(".")[0])

    # Adding labels and title
    ax.set_xlabel(avg_duration_per_length[0].index.name)
    ax.set_ylabel("Duration")
    ax.set_xticks(x, avg_duration_per_length[0].index)
    ax.set_title(f"Histogram of Durations over {avg_duration_per_length[0].index.name}")
    ax.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":

    tts_latency_folder = "tts_latency/"

    all_tts_latencies_files = [f for f in listdir(tts_latency_folder) if isfile(join(tts_latency_folder, f))]
    print(all_tts_latencies_files)

    dict_latencies = {}

    for file in all_tts_latencies_files:
        path = tts_latency_folder + file
        df = pd.read_csv(path)
        dict_latencies[file] = df

    # global average duration
    avg_duration = [dict_latencies[model].loc[:, "duration"].mean() for model in dict_latencies]
    print(f"\navg_duration = {avg_duration}")
    # plot_hist_width_consistent(dict_latencies, avg_duration)

    # avg_duration_per_text_type
    avg_duration_per_text_type = [
        dict_latencies[model].groupby("text_type")["duration"].mean() for model in dict_latencies
    ]
    print(f"\navg_duration_per_text_type = {avg_duration_per_text_type}")
    plot_hist_width_consistent_no_ids(dict_latencies, avg_duration_per_text_type)

    # avg_duration_per_length
    avg_duration_per_length = [dict_latencies[model].groupby("nb_words")["duration"].mean() for model in dict_latencies]
    print(f"\navg_duration_per_length = {avg_duration_per_length}")

    plot_hist_width_consistent(dict_latencies, avg_duration_per_length)
