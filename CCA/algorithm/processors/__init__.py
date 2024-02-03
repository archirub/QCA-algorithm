# from os.path import dirname, basename, isfile, join
# import glob

# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [
#     basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
# ]

# from CCA.algorithm.processors import *

from CCA.algorithm.processors.CCA_processor import CCAProcessor
from CCA.algorithm.processors.cells_processor import CellsProcessor
from CCA.algorithm.processors.neighbours_processor import NeighboursProcessor
from CCA.algorithm.processors.evolution_processor import EvolutionProcessor
from CCA.algorithm.processors.tracks_processor import TracksProcessor
