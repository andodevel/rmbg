from enum import Enum


class TFLogLevel(Enum):
    ALL = '0'  # All
    WARN = '1'  # INFO not printed
    ERROR = '2'  # INFO and WARN not printed
    OFF = '3'  # OFF


CPU_DEVICE = "-1"
