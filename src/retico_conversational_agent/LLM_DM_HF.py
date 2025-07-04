"""
LLM Module
==========

A retico module that provides Natural Language Generation (NLG) using a
Large Language Model (LLM), and handles user interruption.

When a full user sentence (COMMIT SpeechRecognitionIUs) is received from
the ASR (a LLM needs a complete sentence to compute Attention), the
module adds the sentence to the previous dialogue turns stored in the
dialogue history, builds the prompt using the previous turns (and
following a defined template), then uses the prompt to generates a
system answer. IUs are created from the generated words, and are sent
incrementally during the genration. Each new word is ADDED, and if if a
punctuation is encountered (end of clause), the IUs corresponding to the
generated clause are COMMITED. The module records the dialogue history
by saving the dialogue turns from both the user and the agent, it gives
the context of the dialogue to the LLM, which is very important to
maintain a consistent dialogue. Update the dialogue history so that it
doesn't exceed a certain threshold of token size. Put the maximum number
of previous turns in the prompt at each new system sentence generation.
The module stops its generation if it receives the information that the
user started talking (user barge-in/interruption of agent turn). The
interruption information is recognized by an DMIU with a parameter
action="hard_interruption". After an interruption, it aligns the agent
interrupted sentence in dialogue history with the last word spoken by
the agent (these informations are recognized when a TextAlignedAudioIU
with parameter final = False is received after an interruption).
A DMIU "soft_interruption" is also used to prevent from stopping the
generation if the user is only backchanneling : a soft_interruption is
first called when the user starts talking during an agent turn, then if
he continues to speak for a duration greater than a threshold, a
"hard_interruption" is called.

The llama-cpp-python library is used to speed up the LLM inference
(execution in C++).

Inputs : SpeechRecognitionIU, VADTurnAudioIU, TextAlignedAudioIU

Outputs : TurnTextIU

example of the prompt template :
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

import threading
import time
from llama_cpp import Llama, llama_chat_format

import retico_core
from retico_core.text import SpeechRecognitionIU
from retico_core.log_utils import log_exception

from .dialogue_history_hf import DialogueHistoryHf
from .utils import device_definition
from .additional_IUs import (
    VADTurnAudioIU,
    TextAlignedAudioIU,
    TurnTextIU,
    DMIU,
    SpeakerAlignementIU,
)


class LlmDmModuleHf(retico_core.AbstractModule):
    """A retico module that provides Natural Language Generation (NLG) using a
    Large Language Model (LLM), and handles user interruption.

    When a full user sentence (COMMIT SpeechRecognitionIUs) is received from
    the ASR (a LLM needs a complete sentence to compute Attention), the
    module adds the sentence to the previous dialogue turns stored in the
    dialogue history, builds the prompt using the previous turns (and
    following a defined template), then uses the prompt to generates a
    system answer. IUs are created from the generated words, and are sent
    incrementally during the genration. Each new word is ADDED, and if if a
    punctuation is encountered (end of clause), the IUs corresponding to the
    generated clause are COMMITED. The module records the dialogue history
    by saving the dialogue turns from both the user and the agent, it gives
    the context of the dialogue to the LLM, which is very important to
    maintain a consistent dialogue. Update the dialogue history so that it
    doesn't exceed a certain threshold of token size. Put the maximum number
    of previous turns in the prompt at each new system sentence generation.
    The module stops its generation if it receives the information that the
    user started talking (user barge-in/interruption of agent turn). The
    interruption information is recognized by an DMIU with a parameter
    action="hard_interruption". After an interruption, it aligns the agent
    interrupted sentence in dialogue history with the last word spoken by
    the agent (these informations are recognized when a TextAlignedAudioIU
    with parameter final = False is received after an interruption).
    A DMIU "soft_interruption" is also used to prevent from stopping the
    generation if the user is only backchanneling : a soft_interruption is
    first called when the user starts talking during an agent turn, then if
    he continues to speak for a duration greater than a threshold, a
    "hard_interruption" is called.

    The llama-cpp-python library is used to speed up the LLM inference
    (execution in C++).

    Inputs : SpeechRecognitionIU, VADTurnAudioIU, TextAlignedAudioIU, SpeakerAlignementIU, DMIU

    Outputs : TurnTextIU
    """

    @staticmethod
    def name():
        return "LLM DM HF Module"

    @staticmethod
    def description():
        return "A module that provides NLG using a LLM."

    @staticmethod
    def input_ius():
        return [
            SpeechRecognitionIU,
            VADTurnAudioIU,
            TextAlignedAudioIU,
            SpeakerAlignementIU,
            DMIU,
        ]

    @staticmethod
    def output_iu():
        return TurnTextIU

    def __init__(
        self,
        model_path,
        model_repo,
        model_name,
        dialogue_history: DialogueHistoryHf,
        use_chat_completion=False,
        device=None,
        context_size=2000,
        n_gpu_layers=100,
        top_k=40,
        top_p=0.95,
        temp=1.0,
        repeat_penalty=1.1,
        verbose=False,
        incrementality_level="clause",  # turn, sentence, clause, word
        **kwargs,
    ):
        """Initializes the LlamaCppMemoryIncremental Module.

        Args:
            model_path (string): local model instantiation. The path to
                the desired local model weights file
                (my_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf for
                example).
            model_repo (string): HF model instantiation. The path to the
                desired remote hugging face model
                (TheBloke/Mistral-7B-Instruct-v0.2-GGUF for example).
            model_name (string): HF model instantiation. The name of the
                desired remote hugging face model
                (mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            dialogue_history (DialogueHistory): The initialized
                DialogueHistory that will contain the previous user and
                agent turn during the dialogue.
            device (string, optional): the device the module will run on
                (cuda for gpu, or cpu)
            context_size (int, optional): Max number of tokens that the
                total prompt can contain. Defaults to 2000.
            n_gpu_layers (int, optional): Number of model layers you
                want to run on GPU. Take the model nb layers if greater.
                Defaults to 100.
            top_k (int, optional): LLM generation parameter. Defaults to
                40.
            top_p (float, optional): LLM generation parameter. Defaults
                to 0.95.
            temp (float, optional): LLM generation parameter. Defaults
                to 1.0.
            repeat_penalty (float, optional): LLM generation parameter.
                Defaults to 1.1.
            verbose (bool, optional): LLM verbose. Defaults to False.
        """
        super().__init__(**kwargs)

        # model
        self.model = None
        self.model_path = model_path
        self.model_repo = model_repo
        self.model_name = model_name
        self.use_chat_completion = use_chat_completion
        self.context_size = context_size
        self.device = device_definition(device)
        self.n_gpu_layers = 0 if self.device != "cuda" else n_gpu_layers
        self.top_k = top_k
        self.top_p = top_p
        self.temp = temp
        self.repeat_penalty = repeat_penalty
        self.verbose = verbose
        self.chat_formatter = None

        # general
        self.thread_active = False
        self.full_sentence = False
        self.interruption = False
        self.nb_clauses = None
        self.interrupted_speaker_iu = None
        self.which_stop_criteria = None

        # dialogue history
        self.dialogue_history = dialogue_history

        # stop generation conditions
        self.punctuation_text = [".", ",", ";", ":", "!", "?"]  # "..."
        self.last_turn_agent_sentence = None
        self.last_turn_agent_sentence_nb_token = None
        self.last_turn_agent_sentence_turn_id = None
        self.last_turn_last_iu = None

        # incrementality level
        self.incrementality_level = incrementality_level  # turn, sentence, clause, word
        self.end_of_sentence_strings = [".", "!", "?", "..."]
        self.end_of_clause_strings = self.end_of_sentence_strings + [",", ";", ":"]
        match self.incrementality_level:
            case "turn":
                self.check_end_of_incremental_chunk = self.is_word_an_end_of_turn
            case "sentence":
                self.check_end_of_incremental_chunk = self.is_word_an_end_of_sentence
            case "clause":
                self.check_end_of_incremental_chunk = self.is_word_an_end_of_clause
            case "word":
                self.check_end_of_incremental_chunk = lambda x: True

    #######
    # LLM MODULE
    #######

    def apply_chat_template_f(self, messages):
        result = self.chat_formatter(messages=messages)
        prompt = self.model.tokenize(
            result.prompt.encode("utf-8"),
            add_bos=not result.added_special,
            special=True,
        )
        return prompt

    def new_user_sentence(self):
        self.dialogue_history.append_utterance(
            {
                "turn_id": self.current_input[-1].turn_id,
                "role": "user",
                "content": " ".join([iu.payload for iu in self.current_input]),
            }
        )

    def new_agent_sentence(self, agent_sentence, turn_id):
        """Function called to register a new agent sentence into the dialogue
        history (utterances attribute). Calculates the exact token number added
        to the dialogue history (with template prefixes and suffixes).

        Args:
            agent_sentence (string): the new agent sentence to register.
            agent_sentence_nb_tokens (int): the number of token
                corresponding to the agent sentence (without template
                prefixes and suffixes).
        """
        self.dialogue_history.append_utterance(
            {
                "turn_id": turn_id,
                "role": "assistant",
                "content": agent_sentence,
            }
        )

    def interruption_alignment_new_agent_sentence(self, new_agent_sentence):
        """After an interruption, this function will align the sentence stored
        in dialogue history with the last word spoken by the agent.

        This function is triggered if the interrupted speaker IU has
        been received before the module has stored the new agent
        sentence in the dialogue history. If that is not the case, the
        function interruption_alignment_last_agent_sentence is triggered
        instead.

        With the informations stored in self.interrupted_speaker_iu,
        this function will shorten the new_agent_sentence to be aligned
        with the last words spoken by the agent.

        Args:
            new_agent_sentence (string): the utterance generated by the
                LLM, that has been interrupted by the user and needs to
                be aligned.
        """
        utterance = {
            "turn_id": self.interrupted_speaker_iu.turn_id,
            "role": "assistant",
            "content": new_agent_sentence,
        }
        self.dialogue_history.interruption_alignment_new_agent_sentence(
            utterance, self.punctuation_text, self.interrupted_speaker_iu
        )

    def interruption_alignment_last_agent_sentence(self, iu):
        """After an interruption, this function will align the sentence stored
        in dialogue history with the last word spoken by the agent.

        If the turn that the speaker IU is a turn further from the last
        utterance in self.utterances, store the speaker IU in
        self.interrupted_speaker_iu and wait for the sentence generation
        to end, the interruption_alignment_new_agent_sentence function
        will then be called.

        Args:
            iu (AudioTurnIU): The IU received from the Speaker module,
                that correspond to the last IU that has been outputted
                through the speakers.
        """
        # it could be the DM that has that function, because the DM receives the IUs from speaker module too
        self.interrupted_speaker_iu = iu
        self.terminal_logger.debug(
            "interruption alignement LLM",
            cl="trace",
            interrupted_iu_turn_id=iu.turn_id,
            last_turn_id=self.last_turn_agent_sentence_turn_id,
        )
        # check if the sentence we want to align has been stored in self.last_turn_agent_sentence
        if self.last_turn_agent_sentence_turn_id:
            if self.last_turn_agent_sentence_turn_id == iu.turn_id:
                self.interruption_alignment_new_agent_sentence(self.last_turn_agent_sentence)

    def generate_next_sentence(self, prompt_tokens):
        """Generates the agent next sentence from the constructed prompt
        (dialogue scenario, dialogue history, instruct...). At each generated
        token, check is the end of the sentence corresponds to a stopping
        pattern, role pattern, or punctuation. Sends the info to the retico
        Module using the submodule function. Stops the generation if a stopping
        token pattern is encountered (using the
        stop_multiple_utterances_generation as the stopping criteria).

        Args:
            subprocess (function): the function to call during the
                sentence generation to possibly send chunks of sentence
                to the children modules.

        Returns:
            string: Agent new generated sentence. int: nb tokens in new
                agent sentence.
        """

        def stopping_criteria(tokens, logits):
            if tokens[-1] == self.model.token_eos():
                self.which_stop_criteria = "stop_token"
                return True
            elif len(tokens) == self.context_size - self.dialogue_history.context_size:
                self.which_stop_criteria = "max_tokens"
                return True
            elif self.interruption:
                self.which_stop_criteria = "interruption"
                return True
            return False

        # Define the parameters
        self.which_stop_criteria = None
        self.nb_clauses = 0
        self.file_logger.debug("start_process")

        # IMPORTANT : the stop crit is executed after the body of the for loop,
        # which means token here is seen inside the loop before being accessible in stop crit funct
        tokens = []
        for token in self.model.generate(
            prompt_tokens,
            stopping_criteria=stopping_criteria,
            top_k=self.top_k,
            top_p=self.top_p,
            temp=self.temp,
            repeat_penalty=self.repeat_penalty,
        ):
            tokens.append(token)
            self.incremental_iu_sending_hf(token)
        return self.model.detokenize(tokens).decode("utf-8", errors="ignore"), len(tokens)

    def create_chat_completion(self, history):
        """Generates the agent next sentence from the constructed prompt
        (dialogue scenario, dialogue history, instruct...). At each generated
        token, check is the end of the sentence corresponds to a stopping
        pattern, role pattern, or punctuation. Sends the info to the retico
        Module using the submodule function. Stops the generation if a stopping
        token pattern is encountered (using the
        stop_multiple_utterances_generation as the stopping criteria).

        Args:
            subprocess (function): the function to call during the
                sentence generation to possibly send chunks of sentence
                to the children modules.
            top_k (int, optional): _description_. Defaults to 40.
            top_p (float, optional): _description_. Defaults to 0.95.
            temp (float, optional): _description_. Defaults to 1.0.
            repeat_penalty (float, optional): _description_. Defaults to
                1.1.

        Returns:
            string: Agent new generated sentence. int: nb tokens in new
                agent sentence.
        """

        # Define the parameters
        # print("final_prompt = ", history)
        self.nb_clauses = 0
        self.file_logger.debug("start_process")
        self.which_stop_criteria = None

        tokens = []
        for token in self.model.create_chat_completion(
            history,
            stream=True,
            # max_tokens=100,
            top_k=self.top_k,
            top_p=self.top_p,
            temp=self.temp,
            repeat_penalty=self.repeat_penalty,
        ):
            tokens.append(token)
            self.incremental_iu_sending_hf(token)

            if len(tokens) == self.context_size - self.dialogue_history.context_size:
                self.which_stop_criteria = "max_tokens"
                break
            elif self.interruption:
                self.which_stop_criteria = "interruption"
                break

        self.which_stop_criteria = "stop_token" if self.which_stop_criteria == None else self.which_stop_criteria
        return self.model.detokenize(tokens).decode("utf-8", errors="ignore"), len(tokens)

    #######
    # RETICO MODULE
    #######

    def finish_with_punctuation(self, tokens: list[int]):
        """Check if the last token is a punctuation token."""
        return self.model.detokenize([tokens[-1]]).decode("utf-8", errors="ignore") in self.punctuation_text

    def is_token_punctuation(self, token: int):
        """Check if the last token is a punctuation token."""
        return self.model.detokenize([token]).decode("utf-8", errors="ignore") in self.punctuation_text

    def is_word_an_end_of_sentence(self, word: str, token: int):
        """Check if the last token is an end of sentence word."""
        return word in self.end_of_sentence_strings

    def is_word_an_end_of_clause(self, word: str, token: int):
        """Check if the last word is an end of clause word."""
        return word in self.end_of_clause_strings

    def is_word_an_end_of_turn(self, word: str, token: int):
        """Check if token is model EOS token."""
        return token == self.model.token_eos()

    def incremental_iu_sending_hf(self, token):
        new_word = self.model.detokenize([token]).decode("utf-8", errors="ignore")
        last_iu = None if len(self.current_input) == 0 else self.current_input[-1]
        um = retico_core.UpdateMessage()
        output_iu = self.create_iu(
            grounded_in=last_iu,
            text=new_word,
            turn_id=last_iu.turn_id,
            clause_id=self.nb_clauses,
        )
        self.current_output.append(output_iu)
        um.add_iu(output_iu, retico_core.UpdateType.ADD)

        if self.check_end_of_incremental_chunk(new_word, token):
            self.nb_clauses += 1
            for iu in self.current_output:
                self.commit(iu)
                um.add_iu(iu, retico_core.UpdateType.COMMIT)
            # self.file_logger.debug("send_clause")
            # self.terminal_logger.debug("send_clause", cl="trace")
            self.file_logger.debug(f"send_{self.incrementality_level}")
            self.terminal_logger.debug(f"send_{self.incrementality_level}", cl="trace")
            self.current_output = []
        self.append(um)

    # def incremental_iu_sending_hf_2(self, tokens):
    #     new_text = self.model.detokenize(tokens).decode("utf-8", errors="ignore")
    #     um, new_words = retico_core.text.get_text_increment(self, new_text)
    #     last_iu = None if len(self.current_input) == 0 else self.current_input[-1]
    #     for w in new_words:
    #         output_iu = self.create_iu(
    #             grounded_in=last_iu,
    #             text=w,
    #             turn_id=last_iu.turn_id,
    #             clause_id=self.nb_clauses,
    #         )
    #         self.current_output.append(output_iu)
    #         um.add_iu(output_iu, retico_core.UpdateType.ADD)

    #     if self.finish_with_punctuation(tokens):
    #         self.terminal_logger.debug("new_text", new_text=new_text, cl="trace")
    #         self.terminal_logger.debug("new_words", new_words=new_words, cl="trace")
    #         self.terminal_logger.debug(
    #             "new_words LLM DM", cl="trace",
    #             new_words=[self.model.detokenize([t]).decode("utf-8", errors="ignore") for t in tokens],
    #         )
    #         self.nb_clauses += 1
    #         for iu in self.current_output:
    #             self.commit(iu)
    #             um.add_iu(iu, retico_core.UpdateType.COMMIT)
    #         self.file_logger.debug("send_clause")
    #         self.terminal_logger.debug("send_clause", cl="trace")
    #         self.current_output = []
    #     self.append(um)

    def process_incremental(self):
        """Function that calls the submodule LLamaCppMemoryIncremental to
        generates a system answer (text) using the chosen LLM.

        Incremental : Use the subprocess function as a callback function
        for the submodule to call to check if the current chunk of
        generated sentence has to be sent to the Children Modules (TTS
        for example).
        """

        # TODO : find a way to have only one data buffer for generated token/text. currently we have competitively IU buffer (current_output), and text buffer (agent_sentence).
        # this way, we would only have to remove from one buffer when deleting stop pattern, or role pattern.
        self.new_user_sentence()
        prompt_tokens, history = self.dialogue_history.prepare_dialogue_history(self.apply_chat_template_f)
        if self.use_chat_completion:
            agent_sentence, agent_sentence_nb_tokens = self.create_chat_completion(history)
        else:
            agent_sentence, agent_sentence_nb_tokens = self.generate_next_sentence(prompt_tokens)

        um = retico_core.UpdateMessage()
        # print("self.which_stop_criteria =", self.which_stop_criteria)
        if self.which_stop_criteria == "interruption":
            self.terminal_logger.debug("interruption", cl="trace")
            # REVOKE every word in interrupted clause (every IU in current_output)
            for iu in self.current_output:
                self.revoke(iu, remove_revoked=False)
                um.add_iu(iu, retico_core.UpdateType.REVOKE)

            # align dialogue history with last word spoken by speaker module
            if self.interrupted_speaker_iu is not None:
                self.interruption_alignment_new_agent_sentence(agent_sentence)

        elif self.which_stop_criteria == "stop_token":
            # add an IU significating that the agent turn is complete (EOT)
            last_processed_iu = self.current_input[-1]
            iu = self.create_iu(
                grounded_in=last_processed_iu,
                final=True,
                turn_id=last_processed_iu.turn_id,
            )
            um.add_iu(iu, retico_core.UpdateType.COMMIT)
            self.terminal_logger.debug(
                "stop_token",
                cl="trace",
            )
            self.last_turn_agent_sentence = agent_sentence
            self.last_turn_agent_sentence_nb_token = agent_sentence_nb_tokens
            self.last_turn_agent_sentence_turn_id = last_processed_iu.turn_id

        elif self.which_stop_criteria == "max_tokens":
            # add an IU significating that the agent turn is complete (EOT)
            last_processed_iu = self.current_input[-1]
            iu = self.create_iu(
                grounded_in=last_processed_iu,
                final=True,
                turn_id=last_processed_iu.turn_id,
            )
            um.add_iu(iu, retico_core.UpdateType.COMMIT)
            self.terminal_logger.debug(
                "stop_token",
                cl="trace",
            )
            self.last_turn_agent_sentence = agent_sentence
            self.last_turn_agent_sentence_nb_token = agent_sentence_nb_tokens
            self.last_turn_agent_sentence_turn_id = last_processed_iu.turn_id

        else:
            raise NotImplementedError("this which_stop_criteria has not been implemented")

        # print(f"LLM:\n{self.last_turn_agent_sentence}")
        self.terminal_logger.info(f"LLM:\n{self.last_turn_agent_sentence}")

        self.terminal_logger.debug(
            "EOT STOP CRIT",
            len_um=len(um),
            finals=[iu.final for iu, _ in um],
            cl="trace",
        )
        self.append(um)

        # reset because it is end of sentence
        # TODO : loop over these 2 list and remove only IUs with the same turn_id as current_turn ? to be sure to keep IUs from next turn ?
        self.current_output = []
        self.current_input = []

    def process_update(self, update_message):
        """Overrides AbstractModule : https://github.com/retico-team/retico-
        core/blob/main/retico_core/abstract.py#L402.

        Args:
            update_message (UpdateType): UpdateMessage that contains new
                IUs, if the IUs are COMMIT, it means that these IUs
                correspond to a complete sentence. All COMMIT IUs (msg)
                are processed calling the process_incremental function.

        Returns:
            _type_: returns None if update message is None.
        """
        if not update_message:
            return None
        msg = []
        # self.last_turn_last_iu = None

        for iu, ut in update_message:
            if isinstance(iu, SpeechRecognitionIU):
                if ut == retico_core.UpdateType.ADD:
                    continue
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    msg.append(iu)
            elif isinstance(iu, DMIU):
                if iu.action == "hard_interruption":
                    self.interruption = True
                    self.file_logger.debug("hard_interruption")
                elif iu.action == "soft_interruption":
                    self.file_logger.debug("soft_interruption")
                elif iu.action == "stop_turn_id":
                    if (
                        len(self.current_output) > 0
                    ):  # do we keep this ? we could interrupt even if it is empty by keeping track of last outputted iu
                        self.terminal_logger.debug("STOP TURN ID", cl="trace")
                        self.file_logger.debug("stop_turn_id")
                        if iu.turn_id > self.current_output[-1].turn_id:
                            self.interruption = True  # test this
                            # we would have to do something much more simple, just stop generation and clear current_output, no alignement or nothing
            elif isinstance(iu, SpeakerAlignementIU):
                # self.terminal_logger.debug("LLM receives SpeakerAlignementIU", cl="trace")
                if ut == retico_core.UpdateType.ADD:
                    if iu.event == "interruption":
                        self.interruption_alignment_last_agent_sentence(iu)
                    if iu.event == "agent_EOT":
                        self.file_logger.debug("LLM agent_EOT")
                        self.new_agent_sentence(
                            self.last_turn_agent_sentence,
                            self.last_turn_last_iu.turn_id,
                        )
                    if iu.event == "ius_from_last_turn":
                        self.last_turn_last_iu = iu
                        # print("LLM IU TURN EOT, ius_from_last_turn", self.last_turn_last_iu)
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue

        if len(msg) > 0:
            self.current_input.extend(msg)
            self.full_sentence = True
            self.interruption = False

    def _llm_thread(self):
        """Function running the LLM, executed ina separated thread so that the
        generation can be interrupted, if the user starts talking (the
        reception of an interruption VADTurnAudioIU)."""
        while self.thread_active:
            try:
                time.sleep(0.01)
                if self.full_sentence:
                    self.terminal_logger.debug("start_answer_generation", cl="trace")
                    self.file_logger.debug("start_answer_generation")
                    self.process_incremental()
                    self.file_logger.debug("EOT")
                    self.full_sentence = False
            except Exception as e:
                log_exception(module=self, exception=e)

    def setup(self, **kwargs):
        """Overrides AbstractModule : https://github.com/retico-team/retico-
        core/blob/main/retico_core/abstract.py#L402.

        Instantiate the model with the given model info, if insufficient
        info given, raise an NotImplementedError. Init the prompt with
        the initialize_prompt function. Calculates the stopping with the
        init_stop_criteria function.
        """
        super().setup(**kwargs)

        if self.model_path is not None:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )

        elif self.model_repo is not None and self.model_name is not None:
            self.model = Llama.from_pretrained(
                repo_id=self.model_repo,
                filename=self.model_name,
                device_map=self.device,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )

        else:
            raise NotImplementedError(
                "Please, when creating the module, you must give a model_path or model_repo and model_name"
            )

        self.chat_formatter = (
            (
                self.model.chat_handler
                or self.model._chat_handlers.get(self.model.chat_format)
                or llama_chat_format.get_chat_completion_handler(self.model.chat_format)
            )
            .__closure__[0]
            .cell_contents
        )

    def prepare_run(self):
        """Overrides AbstractModule : https://github.com/retico-team/retico-
        core/blob/main/retico_core/abstract.py#L808."""
        super().prepare_run()
        self.thread_active = True
        threading.Thread(target=self._llm_thread).start()

    def shutdown(self):
        """Overrides AbstractModule : https://github.com/retico-team/retico-
        core/blob/main/retico_core/abstract.py#L819."""
        super().shutdown()
        self.thread_active = False
