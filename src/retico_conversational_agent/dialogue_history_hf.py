"""
Dialogue History
================

The dialogue history can be used to store every user and agent previous
turns during a dialogue. You can add data using update its data using
the append_utterance function, update the current turn stored using
prepare_dialogue_history and get the updates prompt using get_prompt.

The DialogueHistory is using a template config file, that you can change
to configure the prefixes, suffixes, roles, for the user, agent, system
prompt and the prompt itself. It is useful because every LLm has a
different prefered template for its prompts.

Example of a prompt with the following config :
{
"user": {
"role": "Child",
"role_sep": ":",
"pre": "",
"suf": "\n\n"
},
"agent": {
"role": "Teacher",
"role_sep": ":",
"pre": "",
"suf": "\n\n"
},
"system_prompt": {
"pre": "<<SYS>>\n",
"suf": "<</SYS>>\n\n"
},
"prompt": {
"pre": "[INST] ",
"suf": "[/INST]"
}
}

prompt = "[INST] <<SYS>>
This is a spoken dialog scenario between a teacher and a 8 years old
child student. The teacher is teaching mathematics to the child student.
As the student is a child, the teacher needs to stay gentle all the
time. Please provide the next valid response for the following
conversation. You play the role of a teacher. Here is the beginning of
the conversation :
<</SYS>>

Child : Hello !

Teacher : Hi! How are you today ?

Child : I am fine, and I can't wait to learn mathematics!

[/INST]
Teacher :"
"""

import time


class DialogueHistoryHf:
    """The dialogue history is where all the sentences from the previvous agent
    and user turns will be stored.

    The LLM, and/or DM will retrieve the history to build the prompt and
    use it to generate the next agent turn.
    """

    def __init__(
        self,
        terminal_logger,
        file_logger=None,
        context_size=2000,
        initial_system_prompt="",
        initial_dh=None,
    ):
        """Initializes the DialogueHistory.

        Args:
            terminal_logger (TerminalLogger): The logger used to print
                events in console.
            file_logger (FileLogger, optional): The logger used to store
                events in a log file.. Defaults to None.
            context_size (int, optional): Max number of tokens that the
                total prompt can contain (LLM context size). Defaults to
                2000. Defaults to 2000.
            initial_system_prompt (str, optional): The initial system
                prompt containing the dialogue scenario and/or
                instructions. Defaults to "".
        """
        self.terminal_logger = terminal_logger
        self.file_logger = file_logger
        self.cpt_0 = 1
        self.context_size = context_size
        # with open(prompt_format_config_file, "r", encoding="utf-8") as config:
        #     self.prompt_format_config = json.load(config)
        if initial_dh is not None:
            self.dialogue_history = initial_dh
            if initial_dh[0]["role"] == "system_prompt":
                self.initial_system_prompt = initial_dh[0]["content"]
                self.current_system_prompt = initial_dh[0]["content"]
        else:
            self.initial_system_prompt = initial_system_prompt
            self.current_system_prompt = initial_system_prompt
            self.dialogue_history = [
                {
                    "turn_id": -1,
                    "role": "system_prompt",
                    "content": initial_system_prompt,
                }
            ]

    # Setters

    def append_utterance(self, utterance):
        """Add the utterance to the dialogue history.

        Args:
            utterance (dict): a dict containing the speaker and the
                turn's transcription (text of the sentences).
        """
        assert set(("turn_id", "role", "content")) <= set(utterance)
        # insure that turn_id is not None, and increment turn_id for system that do not have a turn id cpt (like DM).
        utterance["turn_id"] = len(self.dialogue_history) if utterance["turn_id"] is None else utterance["turn_id"]
        self.dialogue_history.append(utterance)
        print(f"\n{utterance['role']} : {utterance['content']}")
        time.sleep(5)

    def reset_system_prompt(self):
        """Set the system prompt to initial_system_prompt, which is the prompt
        given at the DialogueHistory initialization."""
        self.change_system_prompt(self.initial_system_prompt)

    def change_system_prompt(self, system_prompt):
        """Function that changes the DialogueHistory current system prompt. The
        system prompt contains the LLM instruction and the scenario of the
        interaction.

        Args:
            system_prompt (str): the new system_prompt.

        Returns:
            str: the previous system_prompt.
        """
        previous_system_prompt = self.current_system_prompt
        self.current_system_prompt = system_prompt
        self.dialogue_history[0]["content"] = system_prompt
        return previous_system_prompt

    def get_prompt(self, fun_tokenize, start=0, end=None):
        """Get the formatted prompt containing all turns between start and end.

        Args:
            start (int, optional): start id of the oldest turn to take.
                Defaults to 1.
            end (int, optional): end id of the latest turn to take.
                Defaults to None.

        Returns:
            List[int]: the corresponding formatted prompt tokens.
        """
        if end is None:
            end = len(self.dialogue_history)
        print("start : ", start, " end : ", end)
        if end <= start:
            return []

        if self.dialogue_history[0]["role"] == "system":
            system_prompt = [self.dialogue_history[0]]
        else:
            print("no system prompt")
            system_prompt = []

        dh = system_prompt + self.dialogue_history[start:end]
        return (dh, fun_tokenize(dh))

    def prepare_dialogue_history(self, fun_tokenize):
        """Calculate if the current dialogue history is bigger than the LLM's
        context size (in nb of token). If the dialogue history contains too
        many tokens, remove the older dialogue turns until its size is smaller
        than the context size. The self.cpt_0 class argument is used to store
        the id of the older turn of last prepare_dialogue_history call (to
        start back the while loop at this id).

        Args:
            fun_tokenize (Callable[]): the tokenize function given by
                the LLM, so that the DialogueHistory can calculate the
                right dialogue_history size.

        Returns:
            (List[int]): the prompt tokens to give the LLM.
        """

        dh, prompt_tokens = self.get_prompt(fun_tokenize, start=self.cpt_0)
        nb_tokens = len(prompt_tokens)
        print("nb_tokens : ", nb_tokens, " context_size : ", self.context_size)
        while nb_tokens > self.context_size:
            self.cpt_0 += 2
            if self.cpt_0 >= len(self.dialogue_history):
                raise ValueError("System prompt is too long, please increase the context size or cut system prompt.")
            dh, prompt_tokens = self.get_prompt(fun_tokenize, start=self.cpt_0)
            nb_tokens = len(prompt_tokens)
            print("nb_tokens : ", nb_tokens, " context_size : ", self.context_size)
        return prompt_tokens, dh

    def interruption_alignment_new_agent_sentence(self, utterance, punctuation, interrupted_speaker_iu):
        """After an interruption, this function will align the sentence stored
        in dialogue history with the last word spoken by the agent. With the
        informations stored in interrupted_speaker_iu, this function will
        shorten the utterance to be aligned with the last words spoken by the
        agent.

        Args:
            utterance (dict[str]): the utterance generated by the LLM,
                that has been interrupted by the user and needs to be
                aligned.
            punctuation (list[str]): the id of the punctuation
                marks, calculated by the LLM at initialization.
            interrupted_speaker_iu (IncrementalUnit): the
                SpeakerModule's IncrementalUnit, used to align the agent
                utterance.
        """
        new_agent_sentence = utterance["content"]

        # split the sentence into clauses
        sentence_clauses = []
        old_i = 0
        for i, c in enumerate(new_agent_sentence):
            if c in punctuation or i == len(new_agent_sentence) - 1:
                sentence_clauses.append(new_agent_sentence[old_i : i + 1])
                old_i = i + 1

        # remove all clauses after clause_id (the interrupted clause)
        sentence_clauses = sentence_clauses[: interrupted_speaker_iu.clause_id + 1]

        # Shorten the last agent utterance until the last char outputted by the speakermodule before the interruption
        sentence_clauses[-1] = sentence_clauses[-1][: interrupted_speaker_iu.char_id + 1]

        # Merge the clauses back together
        new_agent_sentence = "".join(sentence_clauses)

        # store the new sentence in the dialogue history
        utterance["content"] = new_agent_sentence
        self.append_utterance(utterance)

        print("INTERRUPTED AGENT SENTENCE : ", new_agent_sentence)

    # Getters

    def get_dialogue_history(self):
        """Get DialogueHistory's dictionary containing the system prompt and
        all previous turns.

        Returns:
            dict: DialogueHistory's dictionary.
        """
        return self.dialogue_history
