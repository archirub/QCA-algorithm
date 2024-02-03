# from os.path import dirname, basename, isfile, join
# import glob

# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [
#     basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
# ]

# from . import *

from CCA.algorithm.evaluators.CCA_evaluator import CCAEvaluator
from CCA.algorithm.evaluators.cells_evaluator import CellsEvaluator
from CCA.algorithm.evaluators.neighbours_evaluator import NeighboursEvaluator
from CCA.algorithm.evaluators.tracks_evaluator import TracksEvaluator
