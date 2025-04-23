import datetime
import os
import time
import pandas as pd
import torch
from llama_cpp import Llama
import traceback

# from dialogue_manager import DialogueHistory

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "../models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

model = Llama(
    model_path=model_path,
    # n_ctx=2000,
    n_ctx=6000,
    n_gpu_layers=100,
    verbose=True,
)

local_path_swb = "swb_hf_chat/"
for data_file in ["train.csv", "dev.csv", "test.csv"]:
    # for data_file in ["test_2.csv"]:
    data = pd.read_csv(local_path_swb + data_file)
# encoding = "utf-8"
# encoding = "ISO-8859-1"
# encoding = "utf-16"

# w_b = model.detokenize([243])
# w_b = model.detokenize([242])
# w = w_b.decode(encoding, errors="ignore")
# print(f"{w_b} : {w}")

# w_b = model.detokenize([244])
# w = w_b.decode(encoding, errors="replace")
# print(f"{w_b} : {w}")

# w_b = model.detokenize([258])
# w = w_b.decode(encoding, errors="replace")
# print(f"{w_b} : {w}")

conv_list = [
    (
        "Person A",
        "okay so um yes we do keep uh well we started out keeping a budget about two years ago we have a computer here at the house and i made a Lotus spreadsheet and went through the year using all of our our checkbook to figure out what we spent each time and whether we were over or under for each month",
    ),
    ("Person B", "uh-huh uh-huh"),
    (
        "Person A",
        "and then basically since then what i've done is is keep track of it through the checkbook so that based on whatever we've got coming in check coming in and how much i'm spending each half of the month and then trying to also spend- and because our house payment is once a month that's our our biggest uh expense so i take half of that amount out of my checkbook each with each paycheck even though it's really still there",
    ),
    ("Person B", "uh-huh uh-huh"),
    ("Person A", "so that i can keep a a good balance running total yeah through the month what do y'all do"),
    (
        "Person B",
        "a running total yeah uh we've we've uh taken how much we have you know write down how much we have coming in each month and then uh we've at the beginning of the year we sat down and determined how much we could spend we sat down- made up different accounts like you know we've set a budget for each you know household expenses or food and clothing and entertainment and then our our own fun money and just stuff like that and then we write down each each time we spend something we write down in a book and end of the month we tally it up to see if how close we've you know we we try to stay within a certain budget so",
    ),
    ("Person A", "um-hum is it is it hard to keep track of it or does it work out pretty well"),
    ("Person B", "um it takes some it takes some dedication to do it but it it works out real well so"),
    ("Person A", "um-hum and and you're staying within your budget and keep everything is working pretty good"),
    ("Person B", "uh-huh yeah yeah i stay within- i have to stay within it so i"),
    ("Person A", "yeah i found-"),
    (
        "Person B",
        "you know and then we have that you know if you can't stay if something comes up and you can't stay within it then we have uh you know a budget for you know like we call our slush fund or something and something- unexpected- unexpected comes up then you're not",
    ),
    ("Person A", "um-hum yeah"),
    ("Person B", "you know you don't feel it so strapped"),
    ("Person A", "you don't have to go out and borrow it somewhere and and do that"),
    (
        "Person B",
        "right yeah because we don't you know we don't charge anything that we can't pay off by the end of the month",
    ),
    (
        "Person A",
        "yeah that's a good choice we've been trying we're trying to uh do that this year we've budgeted the money that we used to spend we were spending on a CODA account with TI and then money we were also buying stock with for that year we've taken that this year and said we're gonna pay off all of our credit cards and uh",
    ),
    ("Person B", "uh-huh <b_aside> you got paper under your table <e_aside> uh-huh"),
    (
        "Person A",
        "we have a another loan with the bank and so we hope by the end of this year that by doing that we'll be free and clear",
    ),
    (
        "Person B",
        "uh-huh to be out of debt free yeah the only thing we have it to pay off is our is a automobile loan and our house payment and that's the only thing we ever we try to stay out of debt so",
    ),
    ("Person A", "yeah that's good to be in that kind of shape what are y'all trying to do long term"),
    (
        "Person B",
        "uh-huh oh as long term we just he has- you know his retirement plan and then to CODA and stuff like that that's all we've and you know we just have our life insurance for right now",
    ),
    ("Person A", "uh-huh"),
    ("Person B", "so we don't have any long term you know in stocks or anything like that right now so"),
    (
        "Person A",
        "yeah mostly what we're doing we've worked we've done the uh CODA account with TI where they we put in so much a month and then they or so much a paycheck and then they match it",
    ),
    ("Person B", "yeah that's what we're doing so"),
    (
        "Person A",
        " and so that that has worked out pretty good and then i used to work for TI and i have when i retired from there or left i took the money that i had in mine and put it in an IRA and we had an out",
    ),
    ("Person B", "yeah uh-huh"),
    (
        "Person A",
        "we had an existing IRA so we have both of us have some money in an IRA that we're also trying to figure to put it we're putting it in CDs right now and then we're also looking at it in possibly getting a mutual fund ",
    ),
    ("Person B", "uh-huh yeah whenever we get enough saved we we stick it in a CD for a while and then uh"),
    ("Person A", " um-hum"),
    ("Person B", "you know and then when we if we need it we wait till it it's expired and then so"),
    (
        "Person A",
        "yeah the other thing that we've done that that was really nice to see we had one of the financial companies um  Hancock- oh John Hancock company came out and their agents did a long term analysis based on salary and uh what we were planning- what  what what our uh goals were on a long term budget in terms of retirement kid's college paying off the house buying a different house ",
    ),
    ("Person B", "uh-huh"),
    (
        "Person A",
        "um special thing buying land and building our own house and they did an analysis for us based on what we were putting in and the time frame that we wanted to look at",
    ),
    ("Person B", "uh-huh"),
    (
        "Person A",
        "and then gave us a good idea back you know some good information back on whether or not we were going to achieve those goals and yeah or not or what we needed to do so that we could achieve them and money we could put in at what time",
    ),
    (
        "Person B",
        "or not yeah uh-huh that sounds interesting we've never done anything- we have you know just our our life insurance guy has come out you know and he's set up uh you know determined how much we need to ",
    ),
    ("Person A", "um-hum"),
    ("Person B", "you know we need if something were to happen"),
    ("Person A", "um-hum yeah that"),
    ("Person B", "you know"),
    (
        "Person A",
        "that's the other financial thing i guess that we've done is with our life insurance is since i'm at home now is is figuring out uh what we would need if something happened to my husband or what he would need if something happened to me",
    ),
    (
        "Person B",
        "yeah right yeah you- you know if i would sell the you know if he something would happen to him i wouldn't stay in Texas i would uh",
    ),
    ("Person A", "that's a a big thing to think about"),
    (
        "Person B",
        "sell the house and move back home you know to my home town and and uh i wouldn't stay here in Texas so",
    ),
    ("Person A", "um-hum yeah"),
    ("Person B", "you know i don't know what he would do but"),
    ("Person A", "okay i guess that's most of my um financial plans right now is is there anything you'd like to add"),
    ("Person B", "yeah mine too nope that's about all for mine"),
    ("Person A", "-okay well it's been nice talking to you"),
    ("Person B", "nice talking to you too bye-bye"),
    ("Person A", "bye-bye"),
]

DH = ""
DH_10 = ""
DH_10_to_20 = []
for i, turn in enumerate(conv_list):
    role, utterance = turn
    DH += f"\n\n{role}: {utterance}"
    if i <= 10:
        DH_10 += f"\n\n{role}: {utterance}"
    if i in range(10, 21):
        DH_10_to_20.append(DH)


prompt_long = f"[INST] <<SYS>>\
This is a spoken dialog scenario between two persons. \
Please provide the next valid response for the following conversation.\
You play the role of Person B. Here is the beginning of the conversation : \
<</SYS>>\
{DH}\
\n\n[/INST]\
\nPerson B:"

prompt_10 = f"[INST] <<SYS>>\
This is a spoken dialog scenario between two persons. \
Please provide the next valid response for the following conversation.\
You play the role of Person B. Here is the beginning of the conversation : \
<</SYS>>\
{DH_10}\
\n\n[/INST]\
\nPerson B:"

prompt_10_to_20 = []
for dh in DH_10_to_20:
    prompt_10_to_20.append(
        f"[INST] <<SYS>>\
        This is a spoken dialog scenario between two persons. \
        Please provide the next valid response for the following conversation.\
        You play the role of Person B. Here is the beginning of the conversation : \
        <</SYS>>\
        {dh}\
        \n\n[/INST]\
        \nPerson B:"
    )

prompt_longer = f"[INST] <<SYS>>\
This is a spoken dialog scenario between two persons. \
Please provide the next valid response for the following conversation.\
You play the role of Person B. Here is the beginning of the conversation : \
<</SYS>>\
{DH+DH+DH}\
\n\n[/INST]\
\nPerson B:"


# prompt = "[INST] <<SYS>>\
# This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
# <</SYS>>\
# \n\nChild: Hello ! \
# \n\nTeacher: Hi! How are your today ?\
# \n\nChild: I am fine !\
# \n\nTeacher:\
# \n\n[/INST]"

punctuation_text = [b".", b",", b";", b":", b"!", b"?", b"..."]
punctuation_ids = [b[0] for b in punctuation_text]


def generate_sentence(prompt, reset=True):
    start_tokenize = datetime.datetime.now()
    prompt_tokens = model.tokenize(bytes(prompt, encoding="utf-8"))
    print("NB tokens = ", len(prompt_tokens))
    end_tokenize = datetime.datetime.now()
    tokenize_duration = end_tokenize - start_tokenize
    # pattern = bytes("\n\nChild:", encoding="utf-8")
    pattern = bytes("\n\n", encoding="utf-8")
    pattern_tokens = model.tokenize(pattern, add_bos=False)

    sentence = b""
    sentence_tokens = []
    # sentence_indiv_detok = []
    # sentence_indiv_decod = []
    generate_duration = None
    first_clause_completed = False
    try:
        start_generate = datetime.datetime.now()
        for t in model.generate(
            prompt_tokens,
            top_k=40,
            top_p=0.95,
            temp=1.0,
            repeat_penalty=1.1,
            reset=reset,
        ):
            sentence_tokens.append(t)
            # method 1
            # # sentence_indiv_detok.append(model.detokenize([t]))
            # # sentence_indiv_decod.append(sentence_indiv_detok[-1].decode("utf-8"))
            # sentence = model.detokenize(sentence_tokens)

            # method 2
            word_bytes = model.detokenize([t])
            word = word_bytes.decode(
                "utf-8", errors="ignore"
            )  # special tokens like 243 can raise an error, so we decide to ignore
            sentence += word_bytes

            if not first_clause_completed:

                if word_bytes in punctuation_text:
                    first_clause_completed = True
                    end_first_clause = datetime.datetime.now()
                    first_clause_duration = end_first_clause - start_generate

            if pattern_tokens == sentence_tokens[-len(pattern_tokens) :]:
                break
            if pattern == sentence[-len(pattern) :]:
                break
        end_generate = datetime.datetime.now()
        generate_duration = end_generate - start_generate

        print(f"prompt = {prompt}")
        print(
            f"DURATIONS = \ntokenize : {tokenize_duration.total_seconds()} \ngenerate : {generate_duration.total_seconds()} \nfirst_clause : {first_clause_duration.total_seconds()}"
        )
    # except Exception as e:
    # print(e)
    # print(e.with_traceback())
    except Exception:
        print(traceback.format_exc())

    sentence_1 = model.detokenize(sentence_tokens).decode("utf-8")
    # sentence_2 = b"".join(sentence_indiv_detok).decode("utf-8")
    # sentence_3 = "".join(sentence_indiv_decod)
    # print(sentence_tokens)
    # print(sentence_indiv_detok)
    # print(sentence_indiv_decod)
    print(f"\n\n sentence : {sentence_1}")

    return (
        tokenize_duration.total_seconds(),
        generate_duration.total_seconds() if generate_duration is not None else 0,
        first_clause_duration.total_seconds() if generate_duration is not None else 0,
    )


p = [prompt_10, prompt_long, prompt_longer]
# p = [prompt_10]
# p = prompt_10_to_20
durations = [[] for i in range(len(p))]
nb_it = 1
for e in range(nb_it):
    for i, prompt in enumerate(p):
        print(prompt)
        durations[i].append(generate_sentence(prompt))

print(durations)
print(
    f"mean durations :\
    \nDH_10 : {sum([d[1] for d in durations[0]])/len(durations[0])}\
    \nDH_long : {sum([d[1] for d in durations[1]])/len(durations[1])}\
    \nDH_longer : {sum([d[1] for d in durations[2]])/len(durations[1])}"
)
print(
    f"mean durations first clause :\
    \nDH_10 : {sum([d[2] for d in durations[0]])/len(durations[0])}\
    \nDH_long : {sum([d[2] for d in durations[1]])/len(durations[1])}\
    \nDH_longer : {sum([d[2] for d in durations[2]])/len(durations[1])}"
)

import matplotlib.pyplot as plt

x = list(range(nb_it))
print(x)
fig, ax = plt.subplots()
ax.plot(x, [d[1] for d in durations[0]], "blue", label="dialogue_history_10")
ax.plot(x, [d[2] for d in durations[0]], "blue", alpha=0.2, label="dialogue_history_10 first_clause")
ax.plot(x, [d[1] for d in durations[1]], "brown", label="full_dialogue_history")
ax.plot(x, [d[2] for d in durations[1]], "brown", alpha=0.2, label="full_dialogue_history first_clause")
ax.plot(x, [d[1] for d in durations[2]], "forestgreen", label="dialogue_longer_history")
ax.plot(x, [d[2] for d in durations[2]], "forestgreen", alpha=0.2, label="dialogue_longer_history first_clause")
ax.set_xlabel("id run")
ax.set_ylabel("duration (seconds)")
ax.legend()
plt.show()
# prompt = b"[INST] <<SYS>>\
# This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
# <</SYS>>\
# \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
# Child : Hello ! \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics ! \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
#     This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation : \
# [/INST]"

# # print(prompt)

# # starttime = time.time()
# # startdate = datetime.datetime.now()
# # tokens = model.tokenize(prompt, add_bos=False)
# # endtime = time.time()
# # enddate = datetime.datetime.now()
# # print("time = ", endtime - starttime)
# # print("datetime = ", enddate - startdate)
# # print("len(tokens) = ", len(tokens))

# # starttime = time.time()
# # startdate = datetime.datetime.now()
# # tokens = model.tokenize(prompt, add_bos=False)
# # endtime = time.time()
# # enddate = datetime.datetime.now()
# # print("time = ", endtime - starttime)
# # print("datetime = ", enddate - startdate)
# # print("len(tokens) = ", len(tokens))

# # starttime = time.time()
# # startdate = datetime.datetime.now()
# # tokens = model.tokenize(prompt, add_bos=False)
# # endtime = time.time()
# # enddate = datetime.datetime.now()
# # print("time = ", endtime - starttime)
# # print("datetime = ", enddate - startdate)
# # print("len(tokens) = ", len(tokens))

# pre = b""
# role = b" Child :"
# suf = b"\n\n"
# prompt_0 = b"Hello, what's your name"
# prompt_1 = b" How are you ?"
# prompt_2 = b" Hello, what's your name "

# concat_0 = pre + role + prompt_0 + suf
# concat_1 = pre + role + prompt_1 + suf
# concat_2 = pre + role + prompt_2 + suf

# nb_tokens_pre = len(model.tokenize(pre, add_bos=False))
# nb_tokens_role = len(model.tokenize(role, add_bos=False))
# nb_tokens_suf = len(model.tokenize(suf, add_bos=False))

# nb_tokens_prompt_0 = len(model.tokenize(prompt_0, add_bos=False))
# nb_tokens_prompt_1 = len(model.tokenize(prompt_1, add_bos=False))
# nb_tokens_prompt_2 = len(model.tokenize(prompt_2, add_bos=False))

# nb_tokens_concat_0 = len(model.tokenize(concat_0, add_bos=False))
# nb_tokens_concat_1 = len(model.tokenize(concat_1, add_bos=False))
# nb_tokens_concat_2 = len(model.tokenize(concat_2, add_bos=False))

# print(
#     f"nb tokens concat 0 : {concat_0} : {nb_tokens_pre} + {nb_tokens_role} + {nb_tokens_prompt_0} + {nb_tokens_suf} = {nb_tokens_pre+nb_tokens_role+nb_tokens_prompt_0+nb_tokens_suf} = {nb_tokens_concat_0} "
# )

# print(
#     f"nb tokens concat 1 : {concat_1} : {nb_tokens_pre} + {nb_tokens_role} + {nb_tokens_prompt_1} + {nb_tokens_suf} = {nb_tokens_pre+nb_tokens_role+nb_tokens_prompt_1+nb_tokens_suf} = {nb_tokens_concat_1} "
# )

# print(
#     f"nb tokens concat 0 : {concat_2} : {nb_tokens_pre} + {nb_tokens_role} + {nb_tokens_prompt_2} + {nb_tokens_suf} = {nb_tokens_pre+nb_tokens_role+nb_tokens_prompt_2+nb_tokens_suf} = {nb_tokens_concat_2} "
# )

# print(model.tokenize(pre, add_bos=False, special=True))
# print(model.tokenize(role, add_bos=False, special=True))
# print(model.tokenize(prompt_1, add_bos=False, special=True))
# print(model.tokenize(suf, add_bos=False, special=True))
# print(model.tokenize(concat_1, add_bos=False, special=True))


# prompt = "[INST] <<SYS>>\
# This is a spoken dialog scenario between a teacher and a 8 years old child student.\
# The teacher is teaching mathematics to the child student.\
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
# You play the role of a teacher. Here is the beginning of the conversation :\
# <</SYS>>\
# Child : Hello !\n\n\
# Teacher : Hi! How are your today ?\n\n\
# Child : I am fine, and I can't wait to learn mathematics !\n\n\
# [/INST]"

# sentence_0 = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation. \
# You play the role of a teacher. Here is the beginning of the conversation :"
# sentence_1 = "Hello ! How are you to..."
# sentence_2 = "Hi!"
# sentence_3 = "Hello ! How are you today ? I can't wait to teach you mathematics !"


# dialogue_history = DialogueHistory(
#     "prompt_format_config.json", initial_system_prompt=sentence_0, context_size=100
# )
# dialogue_history.append_utterance(
#     {"turn_id": 1, "text": sentence_1, "speaker": "agent"}
# )
# dialogue_history.append_utterance({"turn_id": 2, "text": sentence_2, "speaker": "user"})
# # dialogue_history.append_utterance({"turn_id": 3, "text": sentence_3, "speaker": "user"})
# dialogue_history.prepare_dialogue_history(model.tokenize)


# ###########

# ### REPEAT

# print("\n\nREPEAT\n\n")

# print(dialogue_history.get_prompt())

# repeat_system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. \
# You play the role of a teacher, and your last sentence has been interrupted by the child, please repeat the last teacher sentence. \
# Here is the beginning of the conversation :"
# system_prompt = dialogue_history.change_system_prompt(repeat_system_prompt)
# prompt = dialogue_history.get_prompt()
# print(prompt)

# # prompt = "[INST] <<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# # The teacher is teaching mathematics to the child student. \
# # As the student is a child, the teacher needs to stay gentle all the time. \
# # You play the role of a teacher, and your last sentence has been interrupted by the child. Before the interruption you were supposed to say : 'Hello ! How are you today ? I can't wait to teach you mathematics !'. Please provide the next teacher sentence by intergrating the interrupted sentence into your next sentence. \
# # Here is the beginning of the conversation, before the interruption:<</SYS>>\
# # \n\nTeacher: Hello ! How are you to...\
# # \n\nChild: Hi![/INST]"


# # prompt_tokens = model.tokenize(bytes(prompt, encoding="utf-8"))
# # print(prompt)

# # pattern = bytes("\n\n", encoding="utf-8")
# # pattern_tokens = model.tokenize(pattern, add_bos=False)
# # print(pattern_tokens)
# # print(model.tokenize(bytes("\n\nChild:", encoding="utf-8"), add_bos=False))
# # print(model.tokenize(bytes("Child:", encoding="utf-8"), add_bos=False))
# # print(model.tokenize(bytes("Child: ", encoding="utf-8"), add_bos=False))

# # # prompt, prompt_tokens = dialogue_history.prepare_dialogue_history(model.tokenize)
# # # print(prompt)


# # sentence_tokens = []
# # try:
# #     for t in model.generate(
# #         prompt_tokens,
# #         top_k=40,
# #         top_p=0.95,
# #         temp=1.0,
# #         repeat_penalty=1.1,
# #     ):
# #         sentence_tokens.append(t)
# #         sentence = model.detokenize(sentence_tokens)
# #         if sentence_tokens[-4:-2] == [13, 13]:
# #             print(len(sentence_tokens))
# #             print(-len(pattern_tokens))
# #             print(len(sentence))
# #             print(-len(pattern))
# #             print(sentence_tokens[-len(pattern_tokens) :])
# #             print(sentence[-len(pattern) :])
# #         if pattern_tokens == sentence_tokens[-len(pattern_tokens) :]:
# #             break
# #         if pattern == sentence[-len(pattern) :]:
# #             break
# # except Exception as e:
# #     print(e)

# # print(sentence_tokens)
# # sentence = model.detokenize(sentence_tokens)

# # print(sentence)
# # dialogue_history.append_utterance(
# #     {
# #         "turn_id": 5,
# #         "speaker": "agent",
# #         "text": sentence.decode("utf-8"),
# #     }
# # )
# # dialogue_history.change_system_prompt(system_prompt)
# # prompt = dialogue_history.get_prompt()
# # # prompt, prompt_tokens = dialogue_history.prepare_dialogue_history(model.tokenize)

# # print(prompt)


# ### RELANCE
# # prompt = "[INST] <<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# # The teacher is teaching mathematics to the child student. \
# # As the student is a child, the teacher needs to stay gentle all the time. \
# # You play the role of a teacher, and your last sentence 'Great! Are you ready to learn mathematics ?' had no answer from the child. Please provide a next teacher sentence that would re-engage the child in the conversation. \
# # Here is the beginning of the conversation : <</SYS>>\
# # \n\nTeacher: Hello ! How are you today ?\
# # \n\nChild: Hi! I am fine.\
# # \n\nTeacher: Great! Are you ready to learn mathematics ? [/INST]"
# prompt = "[INST] <<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. \
# You play the role of a teacher, and your last sentence 'Great, so what is the addition of two and two ?' had no answer from the child. Please provide a next teacher sentence that would re-engage the child in the conversation. \
# Here is the beginning of the conversation : <</SYS>>\
# \n\nTeacher: Addition is when you add two number together to have a bigger number. For example, the addition of one and two is three. Do you understand ?\
# \n\nChild: Yes I think I do.\
# \n\nTeacher: Great, so what is the addition of two and two ? \
# \n\nChild: ... [/INST]"
# prompt_tokens = model.tokenize(bytes(prompt, encoding="utf-8"))
# pattern = bytes("\n\n", encoding="utf-8")
# pattern_tokens = model.tokenize(pattern, add_bos=False)

# sentence_tokens = []
# try:
#     for t in model.generate(
#         prompt_tokens,
#         top_k=40,
#         top_p=0.95,
#         temp=1.0,
#         repeat_penalty=1.1,
#     ):
#         sentence_tokens.append(t)
#         sentence = model.detokenize(sentence_tokens)
#         if pattern_tokens == sentence_tokens[-len(pattern_tokens) :]:
#             break
#         if pattern == sentence[-len(pattern) :]:
#             break
# except Exception as e:
#     print(e)

# sentence = model.detokenize(sentence_tokens)
# print(sentence)
