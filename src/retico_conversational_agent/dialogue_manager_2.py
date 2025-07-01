from functools import partial
import random
import time
from transitions import Machine

import retico_core

from .dialogue_history import DialogueHistory
from .additional_IUs import (
    VADTurnAudioIU,
    DMIU,
    VADIU,
    SpeakerAlignementIU,
    SpeechRecognitionTurnIU,
)


class DialogueManagerModule_2(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "DialogueManager Module"

    @staticmethod
    def description():
        return "a module that manage the dialogue"

    @staticmethod
    def input_ius():
        return [VADIU, SpeakerAlignementIU]

    @staticmethod
    def output_iu():
        return retico_core.IncrementalUnit

    def __init__(
        self,
        dialogue_history: DialogueHistory,
        silence_dur=1,
        bot_dur=0.4,
        silence_threshold=0.75,
        input_framerate=None,
        incrementality_level="sentence",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_framerate = input_framerate
        self.channels = None
        self.sample_width = None
        self.nframes = None
        self.silence_dur = silence_dur
        self.bot_dur = bot_dur
        self.silence_threshold = silence_threshold
        self._n_sil_audio_chunks = None
        self._n_bot_audio_chunks = None
        self.buffer_pointer = 0
        self.dialogue_state = "opening"
        self.turn_id = 0
        self.repeat_timer = float("inf")
        self.overlap_timer = -float("inf")
        self.turn_beginning_timer = -float("inf")
        self.dialogue_history = dialogue_history
        self.incrementality_level = incrementality_level

        self.fsm = Machine(
            model=self,
            states=[
                "user_speaking",
                "agent_speaking",
                "silence_after_user",
                "silence_after_agent",
                "user_overlaps_agent",
                "agent_overlaps_user",
                "mutual_overlap",
            ],
            initial="silence_after_agent",
        )

        # set the log_transi callback to all non reflexive transitions
        self.log_transis()

        # user_speaking
        self.add_transition_callback(
            "user_speaking",
            "silence_after_user",
            callbacks=[
                self.dialogue_history.reset_system_prompt,
                partial(self.send_event, "user_EOT"),
                partial(self.send_action, action="start_answer_generation"),
                partial(self.send_audio_ius, final=True),
            ],
        )

        self.add_transition_callback(
            "agent_speaking",
            "agent_speaking",
            callbacks=[self.update_current_input],
        )
        self.add_transition_callback(
            "agent_speaking",
            "silence_after_agent",
            callbacks=[partial(self.send_event, "agent_EOT")],
        )
        self.add_transition_callback(
            "agent_speaking",
            "user_overlaps_agent",
            callbacks=[
                self.increment_turn_id,
                partial(self.send_event, "user_barge_in"),
            ],
        )

        self.add_transition_callback(
            "silence_after_user",
            "silence_after_user",
            callbacks=[self.update_current_input],
        )
        self.add_transition_callback(
            "silence_after_user",
            "agent_speaking",
            callbacks=[
                partial(self.send_event, "agent_BOT_new_turn"),
                self.update_current_input,
            ],
        )
        self.add_transition_callback(
            "silence_after_user",
            "user_speaking",
            callbacks=[self.increment_turn_id],
        )

        self.add_transition_callback(
            "silence_after_agent",
            "silence_after_agent",
            callbacks=[self.increment_turn_id],
        )
        self.add_transition_callback(
            "silence_after_agent",
            "agent_speaking",
            callbacks=[
                partial(self.send_event, "agent_BOT_same_turn"),
                self.update_current_input,
            ],
        )
        self.add_transition_callback(
            "silence_after_agent",
            "user_speaking",
            callbacks=[self.increment_turn_id],
        )

        self.add_transition_callback(
            "agent_overlaps_user",
            "agent_overlaps_user",
            callbacks=[partial(self.send_audio_ius)],
        )
        self.add_transition_callback(
            "agent_overlaps_user",
            "agent_speaking",
            callbacks=[partial(self.send_audio_ius, final=True)],
        )
        self.add_transition_callback(
            "agent_overlaps_user",
            "user_speaking",
            callbacks=[partial(self.send_audio_ius)],
        )
        self.add_transition_callback(
            "agent_overlaps_user",
            "silence_after_agent",
            callbacks=[partial(self.send_audio_ius, final=True)],
        )

        self.add_transition_callback(
            "user_overlaps_agent",
            "user_overlaps_agent",
            callbacks=[partial(self.send_audio_ius)],
        )
        self.add_transition_callback(
            "user_overlaps_agent",
            "agent_speaking",
            callbacks=[partial(self.send_audio_ius, final=True)],
        )
        self.add_transition_callback(
            "user_overlaps_agent",
            "user_speaking",
            callbacks=[partial(self.send_audio_ius)],
        )
        self.add_transition_callback(
            "user_overlaps_agent",
            "silence_after_user",
            callbacks=[partial(self.send_audio_ius, final=True)],
        )

        self.add_transition_callback(
            "mutual_overlap",
            "mutual_overlap",
            callbacks=[partial(self.send_audio_ius)],
        )
        self.add_transition_callback(
            "mutual_overlap",
            "agent_speaking",
            callbacks=[partial(self.send_audio_ius, final=True)],
        )
        self.add_transition_callback(
            "mutual_overlap",
            "user_speaking",
            callbacks=[partial(self.send_audio_ius)],
        )
        self.add_transition_callback(
            "mutual_overlap",
            "silence_after_agent",
            callbacks=[partial(self.send_audio_ius, final=True)],
        )

    def run_FSM(self):
        source_state = self.state
        if source_state == "agent_speaking":
            match (self.recognize_agent_eot(), self.recognize_user_bot()):
                case (True, True):
                    self.trigger("to_silence_after_agent")
                case (True, False):
                    self.trigger("to_silence_after_agent")
                case (False, True):
                    self.trigger("to_user_overlaps_agent")
                case (False, False):
                    self.trigger("to_" + source_state)
        elif source_state == "user_speaking":
            match (self.recognize_agent_bot(), self.recognize_user_eot()):
                case (True, True):
                    self.trigger("to_silence_after_user")
                case (True, False):
                    self.trigger("to_agent_overlaps_user")
                case (False, True):
                    self.trigger("to_silence_after_user")
                case (False, False):
                    self.trigger("to_" + source_state)
        elif source_state in ["silence_after_user", "silence_after_agent"]:
            match (self.recognize_agent_bot(), self.recognize_user_bot()):
                case (True, True):
                    self.trigger("to_mutual_overlap")
                case (True, False):
                    self.trigger("to_agent_speaking")
                case (False, True):
                    self.trigger("to_user_speaking")
                case (False, False):
                    self.trigger("to_" + source_state)
        elif source_state in [
            "user_overlaps_agent",
            "agent_overlaps_user",
            "mutual_overlap",
        ]:
            match (self.recognize_agent_eot(), self.recognize_user_eot()):
                case (True, True):
                    self.trigger("to_silence_after_agent")
                case (True, False):
                    self.trigger("to_agent_speaking")
                case (False, True):
                    self.trigger("to_user_speaking")
                case (False, False):
                    self.trigger("to_" + source_state)

    def log_transi(self, source, dest):
        self.terminal_logger.info(
            f"switch state {source} -> {dest}",
            cl="trace",
            turn_id=self.turn_id,
        )

    def log_transis(self):
        for t in self.fsm.get_transitions():
            if t.source != t.dest:
                t.after.append(partial(self.log_transi, t.source, t.dest))

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        current_iu_updated = False
        for iu, ut in update_message:
            if isinstance(iu, VADIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    self.current_input.append(iu)
                    current_iu_updated = True

        if current_iu_updated:
            self.run_FSM()

    def add_transition_callback(self, source, dest, callbacks, cond=[]):
        transitions = self.fsm.get_transitions("to_" + dest, source=source, dest=dest)
        if len(transitions) == 1:
            transitions[0].after.extend(callbacks)
        else:
            self.terminal_logger.error(
                "0 or more than 1 transitions with the exact source, dest and trigger. Add the transition directly, or specify."
            )

    def add_soft_interruption_policy(self):
        self.add_transition_callback(
            "silence_after_user",
            "mutual_overlap",
            [partial(self.send_action, "soft_interruption")],
        )
        self.add_transition_callback(
            "silence_after_agent",
            "mutual_overlap",
            [partial(self.send_action, "soft_interruption")],
        )
        self.add_transition_callback(
            "user_speaking",
            "agent_overlaps_user",
            [partial(self.send_action, "soft_interruption")],
        )
        self.add_transition_callback(
            "agent_speaking",
            "user_overlaps_agent",
            [partial(self.send_action, "soft_interruption")],
        )

    def add_continue_policy(self):
        self.fsm.get_transitions(source="user_speaking", dest="silence_after_user")[0].after = []
        self.add_transition_callback(
            "silence_after_user",
            "mutual_overlap",
            [partial(self.set_overlap_timer)],
        )
        self.add_transition_callback(
            "silence_after_agent",
            "mutual_overlap",
            [partial(self.set_overlap_timer)],
        )
        self.add_transition_callback(
            "agent_speaking",
            "user_overlaps_agent",
            [partial(self.set_overlap_timer)],
        )
        self.add_transition_callback(
            "user_speaking",
            "agent_overlaps_user",
            [partial(self.set_overlap_timer)],
        )
        self.add_transition_callback(
            "user_speaking",
            "silence_after_user",
            [partial(self.check_overlap_timer, 1, "user_speaking")],
        )
        self.add_transition_callback(
            "agent_speaking",
            "silence_after_agent",
            [partial(self.check_overlap_timer, 1, "agent_speaking")],
        )
        self.add_transition_callback(
            "agent_overlaps_user",
            "silence_after_agent",
            [partial(self.check_overlap_timer, 1)],
        )
        self.add_transition_callback(
            "user_overlaps_agent",
            "silence_after_user",
            [partial(self.check_overlap_timer, 1)],
        )
        self.add_transition_callback(
            "mutual_overlap",
            "silence_after_user",
            [partial(self.check_overlap_timer, 1)],
        )
        self.add_transition_callback(
            "mutual_overlap",
            "silence_after_agent",
            [partial(self.check_overlap_timer, 1)],
        )

    def add_hard_interruption_policy(self):
        self.add_transition_callback(
            "silence_after_user",
            "mutual_overlap",
            [partial(self.send_action, "hard_interruption")],
        )
        self.add_transition_callback(
            "silence_after_agent",
            "mutual_overlap",
            [partial(self.send_action, "hard_interruption")],
        )
        self.add_transition_callback(
            "agent_speaking",
            "user_overlaps_agent",
            [partial(self.send_action, "hard_interruption")],
        )
        self.add_transition_callback(
            "user_speaking",
            "agent_overlaps_user",
            [partial(self.send_action, "hard_interruption")],
        )

    def add_repeat_policy(self):
        self.add_transition_callback(
            "silence_after_agent",
            "silence_after_agent",
            [self.check_repeat_timer],
        )
        self.add_transition_callback(
            "agent_speaking",
            "silence_after_agent",
            [partial(self.set_repeat_timer, 5)],
        )
        self.add_transition_callback(
            "mutual_overlap",
            "silence_after_agent",
            [partial(self.set_repeat_timer, 5)],
        )
        self.add_transition_callback(
            "agent_overlaps_user",
            "silence_after_user",
            [partial(self.set_repeat_timer, 5)],
        )
        self.add_transition_callback(
            "user_overlaps_agent",
            "silence_after_agent",
            [partial(self.set_repeat_timer, 5)],
        )

    def add_backchannel_policy(self):
        self.add_transition_callback(
            "user_speaking",
            "user_speaking",
            [self.check_backchannel],
        )

    def check_backchannel(self):
        if random.randint(1, 200) > 199:
            self.send_action("back_channel")

    def get_n_audio_chunks(self, n_chunks_param_name, duration):
        """Returns the number of audio chunks containing speech needed in the audio buffer to have a BOT (beginning of turn)
        (ie. to how many audio_chunk correspond self.bot_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.bot_dur.
        """
        if not getattr(self, n_chunks_param_name):
            if len(self.current_input) == 0:
                return None
            first_iu = self.current_input[0]
            self.input_framerate = first_iu.rate
            self.nframes = first_iu.nframes
            self.sample_width = first_iu.sample_width
            # nb frames in each audio chunk
            nb_frames_chunk = len(first_iu.payload) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.input_framerate
            setattr(self, n_chunks_param_name, int(duration / duration_chunk))
        return getattr(self, n_chunks_param_name)

    def recognize(self, _n_audio_chunks=None, threshold=None, condition=None):
        """Function that will calculate if the VAD consider that the user is talking of a long enough duration to predict a BOT.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.bot_dur==0.4 (seconds),
        It returns True if, across the frames corresponding to the last 400ms second of audio, more than 75% are containing speech.

        Returns:
            boolean : the user BOT prediction
        """
        if not _n_audio_chunks or len(self.current_input) < _n_audio_chunks:
            return False
        _n_audio_chunks = int(_n_audio_chunks)
        speech_counter = sum(1 for iu in self.current_input[-_n_audio_chunks:] if condition(iu))
        if speech_counter >= int(threshold * _n_audio_chunks):
            return True
        return False

    def recognize_user_bot(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(n_chunks_param_name="_n_bot_audio_chunks", duration=self.bot_dur),
            threshold=self.silence_threshold,
            condition=lambda iu: iu.va_user,
        )

    def recognize_user_eot(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(
                n_chunks_param_name="_n_sil_audio_chunks", duration=self.silence_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: not iu.va_user,
        )

    def recognize_agent_bot(self):
        return self.current_input[-1].va_agent

    def recognize_agent_eot(self):
        return not self.current_input[-1].va_agent

    def send_event(self, event):
        """Send message that describes the event that triggered the transition

        Args:
            event (str): event description
        """
        self.terminal_logger.info(f"event = {event}", cl="trace", turn_id=self.turn_id)
        # output_iu = self.create_iu(event=event, turn_id=self.turn_id)
        output_iu = DMIU(
            creator=self,
            iuid=f"{hash(self)}:{self.iu_counter}",
            previous_iu=self._previous_iu,
            event=event,
            turn_id=self.turn_id,
        )
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

    def send_action(self, action):
        """Send message that describes the actions the event implies to perform

        Args:
            action (str): action description
        """
        self.terminal_logger.info(f"action = {action}", cl="trace", turn_id=self.turn_id)
        self.file_logger.info(action)
        # output_iu = self.create_iu(action=action, turn_id=self.turn_id)
        output_iu = DMIU(
            creator=self,
            iuid=f"{hash(self)}:{self.iu_counter}",
            previous_iu=self._previous_iu,
            action=action,
            turn_id=self.turn_id,
        )
        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
        self.append(um)

    def send_audio_ius(self, final=False):
        um = retico_core.UpdateMessage()
        ius = []

        if self.incrementality_level == "audio_iu":
            new_ius = self.current_input[self.buffer_pointer :]
            self.buffer_pointer = len(self.current_input)
            for iu in new_ius:
                output_iu = DMIU(
                    creator=self,
                    iuid=f"{hash(self)}:{self.iu_counter}",
                    previous_iu=self._previous_iu,
                    grounded_in=self.current_input[-1],
                    raw_audio=iu.payload,
                    nframes=self.nframes,
                    rate=self.input_framerate,
                    sample_width=self.sample_width,
                    turn_id=self.turn_id,
                    action="process_audio",
                )
                ius.append((output_iu, retico_core.UpdateType.ADD))

        if final:
            for iu in self.current_input:
                output_iu = DMIU(
                    creator=self,
                    iuid=f"{hash(self)}:{self.iu_counter}",
                    previous_iu=self._previous_iu,
                    grounded_in=self.current_input[-1],
                    raw_audio=iu.payload,
                    nframes=self.nframes,
                    rate=self.input_framerate,
                    sample_width=self.sample_width,
                    turn_id=self.turn_id,
                    action="process_audio",
                )
                ius.append((output_iu, retico_core.UpdateType.COMMIT))
            self.current_input = []
            self.buffer_pointer = 0

        um.add_ius(ius)
        self.append(um)

    def update_current_input(self):
        self.current_input = self.current_input[
            -int(self.get_n_audio_chunks(n_chunks_param_name="_n_bot_audio_chunks", duration=self.bot_dur)) :
        ]

    def increment_turn_id(self):
        self.turn_id += 1

    def set_turn_beginning_timer(self):
        self.turn_beginning_timer = time.time()

    def check_turn_beginning_timer(self, duration_threshold=0.5):
        # if it is the beginning of the turn, set_overlap_timer
        if self.turn_beginning_timer + duration_threshold >= time.time():
            self.terminal_logger.debug(
                "it is the beginning of the turn, set_overlap_timer",
                cl="trace",
            )
            self.set_overlap_timer()
            self.turn_beginning_timer = -float("inf")

    def set_overlap_timer(self):
        self.overlap_timer = time.time()

    def check_overlap_timer(self, duration_threshold=1, source_state=None):
        self.terminal_logger.debug(
            f"overlap duration = {time.time() - self.overlap_timer}",
            cl="trace",
        )
        if self.overlap_timer + duration_threshold >= time.time():
            self.terminal_logger.debug(
                "overlap failed because both user and agent stopped talking, send repeat action to speaker module:",
                cl="trace",
            )
            self.send_action("continue")
            self.overlap_timer = -float("inf")
        else:
            if source_state == "user_speaking":
                self.dialogue_history.reset_system_prompt()
                self.send_event(event="user_EOT")
                self.send_action(action="stop_turn_id")
                self.send_action(action="start_answer_generation")
                self.send_audio_ius(final=True)

    def set_repeat_timer(self, offset=3):
        self.repeat_timer = time.time() + offset

    def reset_repeat_timer(self):
        self.repeat_timer = time.time()

    def check_repeat_timer(self):
        if self.repeat_timer < time.time():
            self.increment_turn_id()
            self.terminal_logger.debug(
                "repeat timer exceeded, send repeat action :",
                cl="trace",
                turn_id=self.turn_id,
            )

            dh = self.dialogue_history.get_dialogue_history()
            last_sentence = dh[-1]["content"]
            repeat_system_prompt = (
                "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
            The teacher is teaching mathematics to the child student. \
            As the student is a child, the teacher needs to stay gentle all the time. \
            You play the role of a teacher, and your last sentence '"
                + last_sentence
                + "' had no answer from the child. Please provide a next teacher sentence that would re-engage the child in the conversation. \
            Here is the beginning of the conversation :"
            )
            previous_system_prompt = self.dialogue_history.change_system_prompt(repeat_system_prompt)
            um = retico_core.UpdateMessage()
            iu = SpeechRecognitionTurnIU(
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                previous_iu=None,
                grounded_in=None,
                predictions=["..."],
                text="...",
                stability=0.0,
                confidence=0.99,
                final=True,
                turn_id=self.turn_id,
            )
            ius = [
                (iu, retico_core.UpdateType.ADD),
                (iu, retico_core.UpdateType.COMMIT),
            ]
            um.add_ius(ius)
            self.append(um)
            self.repeat_timer = float("inf")


class VADTurnModule2(retico_core.AbstractModule):
    """A retico module using webrtcvad's Voice Activity Detection (VAD) to enhance AudioIUs with
    turn-taking informations (like user turn, silence or interruption).
    It takes AudioIUs as input and transform them into VADTurnAudioIUs by adding to it turn-taking
    informations through the IU parameter vad_state.
    It also takes TextAlignedAudioIUs as input (from the SpeakerModule), which provides information
    on when the speakers are outputting audio (when the agent is talking).

    The module considers that the current dialogue state (self.user_turn_text) can either be :
    - the user turn
    - the agent turn
    - a silence between two turns

    The transitions between the 3 dialogue states are defined as following :
    - If, while the dialogue state is a silence and the received AudioIUS are recognized as
    containing speech (VA = True), it considers that dialogue state switches to user turn, and sends
    (ADD) these IUs with vad_state = "user_turn".
    - If, while the dialogue state is user turn and a long silence is recognized (with a defined
    threshold), it considers that it is a user end of turn (EOT). It then COMMITS all IUs
    corresponding to current user turn (with vad_state = "user_turn") and dialogue state switches to
    agent turn.
    - If, while the dialogue state is agent turn, it receives the information that the SpeakerModule
    has outputted the whole agent turn (a TextAlignedAudioIU with final=True), it considers that it
    is an agent end of turn, and dialogue state switches to silence.
    - If, while the dialogue state is agent turn and before receiving an agent EOT from
    SpeakerModule, it recognize audio containing speech, it considers the current agent turn is
    interrupted by the user (user barge-in), and sends this information to the other modules to make
    the agent stop talking (by sending an empty IU with vad_state = "interruption"). Dialogue state
    then switches to user turn.

    Inputs : AudioIU, TextAlignedAudioIU

    Outputs : VADTurnAudioIU
    """

    @staticmethod
    def name():
        return "VADTurn Module"

    @staticmethod
    def description():
        return "a module enhancing AudioIUs with turn-taking states using webrtcvad's VAD"

    @staticmethod
    def input_ius():
        return [VADIU]

    @staticmethod
    def output_iu():
        return VADTurnAudioIU

    def __init__(
        self,
        printing=False,
        silence_dur=1,
        bot_dur=0.4,
        silence_threshold=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.printing = printing
        self.input_framerate = None
        self.channels = None
        self.sample_width = None
        self.nframes = None
        self.silence_dur = silence_dur
        self.bot_dur = bot_dur
        self.silence_threshold = silence_threshold
        self._n_sil_audio_chunks = None
        self._n_bot_audio_chunks = None
        self.vad_state = False
        self.user_turn = False
        self.user_turn_text = "no speaker"
        self.buffer_pointer = 0

    def get_n_audio_chunks(self, n_chunks_param_name, duration):
        """Returns the number of audio chunks containing speech needed in the audio buffer to have a BOT (beginning of turn)
        (ie. to how many audio_chunk correspond self.bot_dur)

        Returns:
            int: the number of audio chunks corresponding to the duration of self.bot_dur.
        """
        if not getattr(self, n_chunks_param_name):
            if len(self.current_input) == 0:
                return None
            first_iu = self.current_input[0]
            self.input_framerate = first_iu.rate
            self.nframes = first_iu.nframes
            self.sample_width = first_iu.sample_width
            # nb frames in each audio chunk
            nb_frames_chunk = len(first_iu.payload) / 2
            # duration of 1 audio chunk
            duration_chunk = nb_frames_chunk / self.input_framerate
            setattr(self, n_chunks_param_name, int(duration / duration_chunk))
        return getattr(self, n_chunks_param_name)

    def recognize(self, _n_audio_chunks=None, threshold=None, condition=None):
        """Function that will calculate if the VAD consider that the user is talking of a long enough duration to predict a BOT.
        Example :
        if self.silence_threshold==0.75 (percentage) and self.bot_dur==0.4 (seconds),
        It returns True if, across the frames corresponding to the last 400ms second of audio, more than 75% are containing speech.

        Returns:
            boolean : the user BOT prediction
        """
        if not _n_audio_chunks or len(self.current_input) < _n_audio_chunks:
            return False
        _n_audio_chunks = int(_n_audio_chunks)
        speech_counter = sum(1 for iu in self.current_input[-_n_audio_chunks:] if condition)
        if speech_counter >= int(threshold * _n_audio_chunks):
            return True
        return False

    def recognize_bot(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(n_chunks_param_name="_n_bot_audio_chunks", duration=self.bot_dur),
            threshold=self.silence_threshold,
            condition=lambda iu: iu.va_user,
        )

    def recognize_silence(self):
        return self.recognize(
            _n_audio_chunks=self.get_n_audio_chunks(
                n_chunks_param_name="_n_sil_audio_chunks", duration=self.silence_dur
            ),
            threshold=self.silence_threshold,
            condition=lambda iu: not iu.va_user,
        )

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        for iu, ut in update_message:
            if isinstance(iu, VADIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    self.current_input.append(iu)

        if self.user_turn_text == "agent":
            # It is not a user turn, The agent could be speaking, or it could have finished speaking.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                # self.user_turn = True
                self.user_turn_text = "user_turn"
                self.buffer_pointer = 0

                output_iu = self.create_iu(
                    grounded_in=self.current_input[-1],
                    vad_state="interruption",
                )
                self.terminal_logger.info("interruption")
                self.file_logger.info("interruption")

                return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

            else:
                # print("SILENCE")
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.current_input = self.current_input[
                    -int(
                        self.get_n_audio_chunks(
                            n_chunks_param_name="_n_bot_audio_chunks",
                            duration=self.bot_dur,
                        )
                    ) :
                ]
                # print("remove from audio buffer")

        # else:
        elif self.user_turn_text == "user_turn":
            # It is user turn, we are listenning for a long enough silence, which would be analyzed as a user EOT.
            silence = self.recognize_silence()
            if not silence:
                # print("TALKING")
                # User was talking, and is still talking
                # no user EOT has been predicted, we'll :
                # - Send all new IUs containing audio corresponding to parts of user sentence to the whisper module to generate a new transcription hypothesis.
                # print("len(self.audio_buffer) = ", len(self.audio_buffer))
                # print("self.buffer_pointer = ", self.buffer_pointer)
                new_ius = self.current_input[self.buffer_pointer :]
                self.buffer_pointer = len(self.current_input)
                ius = []
                for iu in new_ius:
                    output_iu = self.create_iu(
                        grounded_in=self.current_input[-1],
                        raw_audio=iu.payload,
                        nframes=self.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        vad_state="user_turn",
                    )
                    ius.append((output_iu, retico_core.UpdateType.ADD))
                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                return um

            else:
                self.terminal_logger.info("user_EOT")
                self.file_logger.info("user_EOT")
                # User was talking, but is not talking anymore (a >700ms silence has been observed)
                # a user EOT has been predicted, we'll :
                # - ADD additional IUs if there is some (sould not happen)
                # - COMMIT all audio in audio_buffer to generate the transcription from the full user sentence using ASR.
                # - set the user_turn as False
                # - empty the audio buffer
                ius = []

                # Add the last AudioIU if there is additional audio since last update_message (should not happen)
                if self.buffer_pointer != len(self.current_input) - 1:
                    for iu in self.current_input[-self.buffer_pointer :]:
                        output_iu = self.create_iu(
                            grounded_in=self.current_input[-1],
                            raw_audio=iu.payload,
                            nframes=self.nframes,
                            rate=self.input_framerate,
                            sample_width=self.sample_width,
                            vad_state="user_turn",
                        )
                        ius.append((output_iu, retico_core.UpdateType.ADD))

                for iu in self.current_input:
                    output_iu = self.create_iu(
                        grounded_in=self.current_input[-1],
                        raw_audio=iu.payload,
                        nframes=self.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        vad_state="user_turn",
                    )
                    ius.append((output_iu, retico_core.UpdateType.COMMIT))

                um = retico_core.UpdateMessage()
                um.add_ius(ius)
                self.user_turn_text = "agent"
                self.current_input = []
                self.buffer_pointer = 0
                return um

        elif self.user_turn_text == "no speaker":
            # nobody is speaking, we are waiting for user to speak.
            # We are listenning for potential user beginning of turn (bot).
            bot = self.recognize_bot()
            if bot:
                self.terminal_logger.info("user_BOT")
                self.file_logger.info("user_BOT")
                # user wasn't talking, but he starts talking
                # A bot has been detected, we'll :
                # - set the user_turn parameter as True
                # - Take only the end of the audio_buffer, to remove the useless audio
                # - Send a INTERRUPTION IU to all modules to make them stop generating new data (if the agent is talking, he gets interrupted by the user)
                # self.user_turn = True
                self.user_turn_text = "user_turn"
            else:
                # user wasn't talkin, and stays quiet
                # No bot has been detected, we'll
                # - empty the audio buffer to remove useless audio
                self.current_input = self.current_input[
                    -int(
                        self.get_n_audio_chunks(
                            n_chunks_param_name="_n_bot_audio_chunks",
                            duration=self.bot_dur,
                        )
                    ) :
                ]
