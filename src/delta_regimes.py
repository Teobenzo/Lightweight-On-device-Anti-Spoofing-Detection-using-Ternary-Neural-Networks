import numpy as np
from abc import ABC, abstractmethod
import math
from typing import Literal


'''
    Code from: Ternary Neural Networks for Gait Identification in Wearable Devices - Agnetti et al.
'''

class DeltaRegime(ABC):
    name: str

    @abstractmethod
    def get(self, epoch: int) -> float:
        pass


class Zero(DeltaRegime):
    name = "zero"

    def get(self, epoch):
        return 0


class Const(DeltaRegime):
    name = "const"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.max = max
        self.min = max
        self.max_at_epoch = 0
        self.const = max

    def get(self, epoch):
        return self.const


class Linear(DeltaRegime):
    name = "linear"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

    def get(self, epoch):
        perc = min(epoch / self.max_at_epoch, 1)
        return self.min + (self.max - self.min) * perc


class Exponential(DeltaRegime):
    name = "exp"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

        self.k = math.log(1 + self.max - self.min) / max_at_epoch

    def get(self, epoch):
        return min(self.min + math.exp(self.k * epoch) - 1, self.max)


class Sqrt(DeltaRegime):
    name = "sqrt"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

    def get(self, epoch):
        perc = min(math.sqrt(float(epoch) / self.max_at_epoch), 1)
        return self.min + (self.max - self.min) * perc


class Square(DeltaRegime):
    name = "square"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

    def get(self, epoch):
        perc = min(math.pow(float(epoch) / self.max_at_epoch, 2), 1)
        return self.min + (self.max - self.min) * perc


class Log(DeltaRegime):
    name = "log"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

    def get(self, epoch):
        perc = math.log(float(epoch) + 1) / math.log(float(self.max_at_epoch) + 1)
        perc = min(perc, 1)
        return self.min + (self.max - self.min) * perc


all = [Const, Linear, Sqrt, Square, Log]
all_dict = {dr.name: dr for dr in all}
all_names = list(all_dict.keys())


def by_name(name: Literal["const", "linear", "sqrt", "square", "log"]) -> DeltaRegime:
    return all_dict[name]