import time
from typing import List
import os


class TimeItData():
    key: str
    hierarchy_level : int = 0
    start_time_ns : int = 0
    end_time_ns : int = 0
    start_offset_ns : int = 0

    children: List["TimeItData"]

    def __init__(self, name, hierarchy_level):
        self.key = name
        self.hierarchy_level = hierarchy_level
        self.children = []

    def start_time(self):
        self.start_time_ns = time.perf_counter_ns()

    def end_time(self):
        self.end_time_ns = time.perf_counter_ns()


class TimeIt:

    def __init__(self, logdir):
        self.data = TimeItData("time", 0)
        self.file = os.path.join(logdir, "params", "timing.txt")

    def printing_data_handler(self, timing : TimeItData):
        diff = timing.end_time_ns - timing.start_time_ns
        sec = str(int(diff / 1000000000))
        ms = str(int((diff % 1000000000) / 1000000))
        us = str(int((diff % 1000000) / 1000))
        timing_file = open(self.file, 'a+')
        timing_file.write(str(" " * timing.hierarchy_level) + str(timing.key) + ": " + sec + "s " + ms + "ms " + us + "us\n")
        timing_file.close()
        for child_timing in timing.children :
            self.printing_data_handler(child_timing)
        return self.printing_data_handler
