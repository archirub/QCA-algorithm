from .cells_processor import CellsProcessor
from .neighbours_processor import NeighboursProcessor
from .evolution_processor import EvolutionProcessor
from .tracks_processor import TracksProcessor


class CCAProcessor:
    """
    Processor for the Classical CA algorithm for particle track reconstruction
    """

    def __init__(self, hits):
        self.hits = hits
        self.CA = {}

    def runAll(
        self, cell_angle, neigh_angle, min_track_length, printArgs=(True, False)
    ):
        self.formCells(cell_angle, printArgs)
        self.findNeighbours(neigh_angle, printArgs)
        self.evolve(printArgs)
        self.generateTracks(min_track_length, printArgs)

    def formCells(self, min_angle, printArgs=(True, False)):
        self.cellsProcessor = CellsProcessor(self.hits, min_angle)
        self.cellsProcessor.form(*printArgs)

    def findNeighbours(self, max_angle, printArgs=(True, False)):
        if not hasattr(self, "cellsProcessor"):
            print("Must call formCells before finding neighbours.")
            return

        self.neighboursProcessor = NeighboursProcessor(self.cellsProcessor, max_angle)
        self.neighboursProcessor.find_neighbours(*printArgs)
        self.CA = self.neighboursProcessor.CA

    def evolve(self, printArgs=(True, False)):
        if not hasattr(self, "neighboursProcessor"):
            print("Must call findNeighbours before evolving.")
            return

        self.evolutionProcessor = EvolutionProcessor(self.neighboursProcessor)
        self.evolutionProcessor.evolve(*printArgs)
        self.CA = self.evolutionProcessor.CA

    def generateTracks(self, min_length, printArgs=(True, False)):
        if not hasattr(self, "evolutionProcessor"):
            print("Must call evolve before generating tracks.")
            return

        self.tracksProcessor = TracksProcessor(self.evolutionProcessor, min_length)
        self.tracksProcessor.generate(*printArgs)
        self.tracks = self.tracksProcessor.tracks

