from enum import Enum

class TaskResult(Enum):
    SUCCESS = 1
    FAILURE = -1
    CRITICAL_BATTERY = -2

class TOD(Enum):
    STARTING = "STARTING"
    POSTER = "POSTER"
    BUFFET = "BUFFET"
    OFF = "OFF"

TODS = {t.value: i for i, t in enumerate(TOD)}

class WP(Enum):
    A1 = "A1"
    B1 = "B1"
    C1 = "C1"
    D1 = "D1"
    A_L1 = "A-L1"
    A_L2 = "A-L2"
    B_L1 = "B-L1"
    B_L2 = "B-L2"
    C_L1 = "C-L1"
    C_L2 = "C-L2"
    D_L1 = "D-L1"
    D_L2 = "D-L2"
    A2 = "A2"
    B2 = "B2"
    C2 = "C2"
    D2 = "D2"
    A3 = "A3"
    B3 = "B3"
    C3 = "C3"
    D3 = "D3"
    DOOR1 = "door1"
    DOOR2 = "door2"
    OUT1 = "out1"
    OUT2 = "out2"
    OUT3 = "out3"
    OUT4 = "out4"
    OUT5 = "out5"
    OUT6 = "out6"

WPS = {wp.value: i for i, wp in enumerate(WP)}
