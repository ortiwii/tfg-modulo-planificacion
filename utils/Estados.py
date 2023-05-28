from enum import Enum


class States(Enum):
    AS_OFF = 0
    AS_READY = 1
    AS_DRIVING = 2
    AS_EMERGENCY_BRAKE = 3
    AS_FINISHED = 4
