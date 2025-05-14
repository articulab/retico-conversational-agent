import datetime
from functools import partial
import os
from pathlib import Path
import colorama
import retico_core
from retico_core.log_utils import filter_cases
import structlog

###########
# FROM CORE
###########


class TerminalLogger(structlog.BoundLogger):
    """Dectorator / Singleton class of structlog.BoundLogger, that is used to configure / initialize
    once the terminal logger for the whole system."""

    def __new__(cls, filters=None):
        if not hasattr(cls, "instance"):
            print("has no TERMINALLOGGER instance")
            # Define filters for the terminal logs
            if filters is not None:
                log_filters = filters
            else:
                print("default filters")
                log_filters = LOG_FILTERS

            def format_module(obj):
                splitted = str(obj).split(" ")
                if splitted[-1] == "Module":
                    splitted.pop(-1)
                return " ".join(splitted)

            def format_timestamp(obj):
                return str(obj[:-4])

            def format_on_type(obj):
                if isinstance(obj, bool):
                    return " ( " + str(obj) + " ) "
                if isinstance(obj, int):
                    return " | " + str(obj) + " | "
                return " " + str(obj)

            cr = structlog.dev.ConsoleRenderer(
                colors=True,
                columns=[
                    structlog.dev.Column(
                        "timestamp",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=None,
                            value_style=colorama.Style.BRIGHT + colorama.Fore.BLACK,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=format_timestamp,
                        ),
                    ),
                    structlog.dev.Column(
                        "level",
                        structlog.dev.LogLevelColumnFormatter(
                            level_styles={
                                key: colorama.Style.BRIGHT + level
                                for key, level in structlog.dev.ConsoleRenderer.get_default_level_styles().items()
                            },
                            reset_style=colorama.Style.BRIGHT + colorama.Style.RESET_ALL,
                            width=None,
                        ),
                    ),
                    structlog.dev.Column(
                        "module",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=None,
                            value_style=colorama.Fore.YELLOW,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=format_module,
                            width=10,
                        ),
                    ),
                    structlog.dev.Column(
                        "event",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=None,
                            value_style=colorama.Style.BRIGHT + colorama.Fore.WHITE,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                            width=40,
                        ),
                    ),
                    structlog.dev.Column(
                        "",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=colorama.Fore.MAGENTA,
                            value_style=colorama.Style.BRIGHT + colorama.Fore.CYAN,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=format_on_type,
                        ),
                    ),
                ],
            )

            # configure structlog to have a terminal logger
            processors = (
                [
                    structlog.processors.TimeStamper(fmt="%H:%M:%S.%f"),
                    structlog.processors.add_log_level,
                ]
                + log_filters
                + [cr]
            )
            structlog.configure(
                processors=processors,
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            terminal_logger = structlog.get_logger("terminal")

            # log info to cache the logger, using the config's cache_logger_on_first_use parameter
            terminal_logger.info("init terminal logger", debug=True)

            # set the singleton instance
            cls.instance = terminal_logger
        return cls.instance


class FileLogger(structlog.BoundLogger):
    """Dectorator / Singleton class of structlog.BoundLogger, that is used to configure / initialize
    once the file logger for the whole system."""

    def __new__(cls, log_path="logs/run", filters=None):
        if not hasattr(cls, "instance"):
            print("has no FILELOGGER instance")
            # Define filters for the terminal logs
            if filters is not None:
                log_filters = filters
            else:
                print("default filters")
                log_filters = LOG_FILTERS
            # configure structlog to have a file logger
            structlog.configure(
                processors=[
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                ]
                + log_filters
                + [
                    structlog.processors.ExceptionRenderer(),
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.WriteLoggerFactory(file=Path(log_path).open("wt", encoding="utf-8")),
                cache_logger_on_first_use=True,
            )
            file_logger = structlog.get_logger("file_logger")

            # log info to cache the logger, using the config's cache_logger_on_first_use parameter
            file_logger.info("init file logger")

            # set the singleton instance
            cls.instance = file_logger
        return cls.instance


def create_new_log_folder(log_folder):
    """Function that creates a new folder to store the current run's log file. Find the last run's
    number and creates a new log folder with an increment of 1.

    Args:
        log_folder (str): the log_folder path where every run's log folder is stored.

    Returns:
        str: returns the final path of the run's log_file, with a format : logs/run_33/logs.log
    """
    cpt = 0
    log_folder_full_path = log_folder + "_" + str(cpt)
    while os.path.isdir(log_folder_full_path):
        cpt += 1
        log_folder_full_path = log_folder + "_" + str(cpt)
    os.makedirs(log_folder_full_path)
    filepath = log_folder_full_path + "/logs.log"
    return filepath


def configurate_logger(log_path="logs/run", filters=None, f=None):
    """
    Configure structlog's logger and set general logging args (timestamps,
    log level, etc.)

    Args:
        log_path: (str): logs folder's path.
        filters: (list): list of function that filters logs that will be outputted in the terminal.
    """
    log_path = create_new_log_folder(log_path)
    terminal_logger = TerminalLogger(filters=filters)
    terminal_logger.info("test", debug=True)
    file_logger = FileLogger(log_path, filters=f)
    return terminal_logger, file_logger


def filter_all_but_warnings_and_errors(_, __, event_dict):
    """function that filters all log message that is not a warning or an error.

    Args:
        event_dict (dict): the log message's dict, containing all parameters passed during logging.

    Returns:
        dict : returns the log_message's event_dict if it went through the filter.
    """
    cases = [
        [
            (
                "level",
                [
                    "warning",
                    "error",
                ],
            ),
        ],
    ]
    return filter_cases(_, _, event_dict, cases=cases)


def run_tests(terminal_logger, file_logger, data, nb_iterations, cpt: int, tests=[1, 2, 3]):
    durations = []

    if 1 in tests:
        ########
        # TEST 1
        ########
        start_time = datetime.datetime.now()
        for i in range(nb_iterations):
            terminal_logger.info(**data, test=1 + cpt * 3)
            file_logger.info(**data, test=1 + cpt * 3)
        end_time = datetime.datetime.now()
        durations.append(end_time - start_time)

    if 2 in tests:
        # ########
        # # TEST 2
        # ########
        start_time = datetime.datetime.now()
        for i in range(nb_iterations):
            file_logger.info(**data, test=2 + cpt * 3)
        end_time = datetime.datetime.now()
        durations.append(end_time - start_time)

    if 3 in tests:
        # ########
        # # TEST 3
        # ########
        start_time = datetime.datetime.now()
        for i in range(nb_iterations):
            terminal_logger.info(**data, test=3 + cpt * 3)
        end_time = datetime.datetime.now()
        durations.append(end_time - start_time)

    return durations


# LOG_FILTERS = []
# LOG_FILTERS = [filter_all_from_modules]
LOG_FILTERS = [filter_all_but_warnings_and_errors]


########
# TEST
########

log_folder = "tests/log"

filters_none = [
    partial(
        filter_cases,
        cases=[
            [("debug", [True])],
        ],
    )
]
filters_all = None
no_filter = []

data = {
    "event": "This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :",
    "module": "Module 1",
    "integer": 123,
    "float": 42414313.42478,
    "debug": True,
}

nb_iterations = 1000000
durations = []

########
# TEST 0
########
start_time = datetime.datetime.now()
for i in range(nb_iterations):
    continue
end_time = datetime.datetime.now()
durations.append(end_time - start_time)


########
# TEST 1
########
def check():
    if i == -1:
        print("hello")


start_time = datetime.datetime.now()
for i in range(nb_iterations):
    check()
    check()
end_time = datetime.datetime.now()
durations.append(end_time - start_time)

# both filter all
terminal_logger, file_logger = configurate_logger(log_folder, filters=filters_all, f=filters_all)
start_time = datetime.datetime.now()
for i in range(nb_iterations):
    terminal_logger.info(**data)
    file_logger.info(**data)
end_time = datetime.datetime.now()
durations.append(end_time - start_time)
del TerminalLogger.instance
del FileLogger.instance
structlog.reset_defaults()

# # TL filter none
# terminal_logger, file_logger = configurate_logger(log_folder, filters=filters_none)
# durations.extend(
#     run_tests(
#         terminal_logger=terminal_logger,
#         file_logger=file_logger,
#         data=data,
#         nb_iterations=nb_iterations,
#         cpt=0,
#         tests=[3],
#     )
# )
# del TerminalLogger.instance
# del FileLogger.instance
# structlog.reset_defaults()

# # TL filter all
# terminal_logger, file_logger = configurate_logger(log_folder, filters=filters_all)
# durations.extend(
#     run_tests(
#         terminal_logger=terminal_logger,
#         file_logger=file_logger,
#         data=data,
#         nb_iterations=nb_iterations,
#         cpt=0,
#         tests=[3],
#     )
# )
# del TerminalLogger.instance
# del FileLogger.instance
# structlog.reset_defaults()

# # FL filter none
# terminal_logger, file_logger = configurate_logger(log_folder, f=filters_none)
# durations.extend(
#     run_tests(
#         terminal_logger=terminal_logger,
#         file_logger=file_logger,
#         data=data,
#         nb_iterations=nb_iterations,
#         cpt=0,
#         tests=[2],
#     )
# )
# del TerminalLogger.instance
# del FileLogger.instance
# structlog.reset_defaults()

# # FL filter all
# terminal_logger, file_logger = configurate_logger(log_folder, f=filters_all)
# durations.extend(
#     run_tests(
#         terminal_logger=terminal_logger,
#         file_logger=file_logger,
#         data=data,
#         nb_iterations=nb_iterations,
#         cpt=0,
#         tests=[2],
#     )
# )

# del TerminalLogger.instance
# del FileLogger.instance
# structlog.reset_defaults()
# terminal_logger, file_logger = configurate_logger(log_folder, filters=filters_all, f=filters_all)
# durations.extend(
#     run_tests(terminal_logger=terminal_logger, file_logger=file_logger, data=data, nb_iterations=nb_iterations, cpt=1)
# )

# del TerminalLogger.instance
# del FileLogger.instance
# structlog.reset_defaults()
# terminal_logger, file_logger = configurate_logger(log_folder, f=no_filter)
# durations.extend(
#     run_tests(
#         terminal_logger=terminal_logger,
#         file_logger=file_logger,
#         data=data,
#         nb_iterations=nb_iterations,
#         cpt=2,
#         tests=[2],
#     )
# )

# del TerminalLogger.instance
# del FileLogger.instance
# structlog.reset_defaults()
# terminal_logger, file_logger = configurate_logger(log_folder, f=filters_none)
# durations.extend(
#     run_tests(
#         terminal_logger=terminal_logger,
#         file_logger=file_logger,
#         data=data,
#         nb_iterations=nb_iterations,
#         cpt=3,
#         tests=[2],
#     )
# )

for i, duration in enumerate(durations):
    print(f"Duration test {i} : {duration} { duration/nb_iterations}")
