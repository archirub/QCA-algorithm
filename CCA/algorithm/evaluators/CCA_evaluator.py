import pandas as pd
from .cells_evaluator import CellsEvaluator
from .neighbours_evaluator import NeighboursEvaluator
from .tracks_evaluator import TracksEvaluator


class CCAEvaluator:
    def __init__(self, CAProcessor, truth):

        self.processor = CAProcessor
        self.truth = truth

        self.cellsEval = CellsEvaluator(self.truth, self.processor.cellsProcessor)
        self.neighsEval = NeighboursEvaluator(
            self.truth, self.processor.neighboursProcessor
        )
        self.tracksEval = TracksEvaluator(self.truth, self.processor.tracksProcessor)

    def evaluate(self):
        self.cellsEval.evaluate()
        self.neighsEval.evaluate()
        self.tracksEval.evaluate()

    @property
    def performance(self):
        dict_ = {
            "cells": self.cellsEval.performance,
            "neighbours": self.neighsEval.performance,
            "tracks": self.tracksEval.performance,
        }

        return pd.DataFrame.from_dict(dict_, orient="index")

