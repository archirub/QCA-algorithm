from CCA.data.cellular_automaton import CellularAutomaton
import numpy as np


class NeighboursProcessor:
    """
    Processor for the cell's neighbours of the classical CA
    
    
    Has properties:
    - cells, cells_per_layer, cells_positions: from cellsProcessor provided on init
    - max_angle: the maximum angle allowed between two cells for them to be neighbours
    - CA: holds the cellular automaton formed by find_neighbours
    """

    def __init__(self, cellsProcessor, max_angle):
        """
            hits: Hits class instance
            min_angle: (float) minimum cone angle allowed between two hits to form a pair
            """
        self.max_angle = max_angle
        self.CA = CellularAutomaton({})

        self.cells = cellsProcessor.cells
        self.cells_per_layer = cellsProcessor.cells_per_layer
        self.cells_positions = cellsProcessor.cells_positions
        self.hits = cellsProcessor.hits

    def __str__(self):
        return str(self.cells)

    def __repr__(self):
        visualize()
        return repr(self.cells)

    def find_neighbours(self, final_print=True, progress_print=False):
        """
        Finds the neighbours among the cells provided.

        Stores in self.CA an (augmented) dictionary of shape
        {
         ["cell_index1"]: {
             "inner_neighs": [arr of cell_indices],
             "state": integer,
             "hit_positions": 
         },
         ["cell_index2"]: {...},
         ...
        }

        In which:
        - cell indices are the indices of the cell in the array "cells"
        - "state" is a property used in the evolution of the  cellular automaton (here, all init to 1)
        - "inner_neighs" are neighbouring cells 
        """
        print("Finding Neighbours...")

        layer_ids = self.hits.layer_ids
        hit_ids = self.hits.hit_ids

        # will store the neighbours and state of the cells
        CA = {
            cell_index: {"state": 1, "inner_neighs": []}
            for cell_index in range(self.cells.shape[0])
        }

        neighs = []

        # Looping over all layers (except the 1st and last)
        for i in range(1, len(layer_ids) - 1):
            if progress_print:
                print("Processing {0}/{1} layer triad.".format(i, len(layer_ids) - 2))

            inner_layer_id = layer_ids[i - 1]
            mid_layer_id = layer_ids[i]

            inner_cell_bounds = self.cells_per_layer[inner_layer_id]
            outer_cell_bounds = self.cells_per_layer[mid_layer_id]

            inner_cells = self.cells[inner_cell_bounds[0] : inner_cell_bounds[1]]
            outer_cells = self.cells[outer_cell_bounds[0] : outer_cell_bounds[1]]

            # STEP 1: CHECK ALL AT ONCE WHICH CELLS ARE CONNECTED (i.e. which inner cell has the same outer hit
            # as the outer cell's inner hit)

            # getting the outer hit of the inner cell, and the inner hit of the outer cell
            hits_inner = inner_cells[:, 1]
            hits_outer = outer_cells[:, 0]

            # Creating a meshgrid of them
            hits_outer_mesh, hits_inner_mesh = np.meshgrid(hits_outer, hits_inner)

            # Checking whether they match all at once
            are_connected = hits_inner_mesh == hits_outer_mesh
            are_connected_indices = np.argwhere(are_connected)

            # Fixing the indexing to reflect index in "cells" (instead of a particular layer in "cells")
            are_connected_indices[:, 0] += inner_cell_bounds[0]
            are_connected_indices[:, 1] += outer_cell_bounds[0]

            #    Note: Now, in "are_neighs_indices"'s 2nd dimension, the 1st location corresponds to the
            #    inner cell, and the 2nd location to the outer cell, for every cells that are neighbours

            # STEP 2: CHECK ALL AT ONCE WHETHER THE ANGLE BETWEEN CONNECTED CELLS LIES BELOW A MAXIMUM ANGLE
            have_ok_angle_indices = self.check_angles(
                are_connected_indices, self.cells_positions, self.max_angle
            )

            # "are_neighs_indices" are the cell pairs that are both connected and have an angle below the threshold
            are_neighs_indices = are_connected_indices[have_ok_angle_indices]

            neighs.extend(are_neighs_indices)

        neighs = np.array(neighs)

        # STEP 3: ADD THE NEIGHBOURS TO "CA"
        for i in range(neighs.shape[0]):
            inner_neigh = neighs[i, 0]
            outer_neigh = neighs[i, 1]

            CA[outer_neigh]["inner_neighs"].append(inner_neigh)

        if final_print:
            print("{0} neighbours found.".format(neighs.shape[0]))
            print("")

        self.CA = CellularAutomaton(CA)

    def check_angles(self, cell_pairs, cells_positions, max_angle):
        """
        Checks whether the angle between each pair of cells is below "max_angle" (gives True if it is)

        - cell_pairs: dimensionality (pair, cell index)
        - max_angle: the maximum allowed angle (float number)
        - cells_positions: array of shape (cell, hit, hit_position_coordinate)

        returns a 1D array of dimensionality (pair)
        """
        #    dimensionality: (connectedPairOfCells, cell (in pair), hit (in cell), position coordinate (of hit))
        positions = cells_positions[cell_pairs]

        # Computing vectors going from inner to outer hit for each cell
        #    dimensionality: (connectedPairOfCells, cell (in pair), position coordinate (of vector))
        vec_diffs = positions[:, :, 1, :] - positions[:, :, 0, :]

        # Computing the magnitude of all vector differences
        #    dimensionality: (connectedPairOfCells, cell (in pair))
        mags = np.linalg.norm(vec_diffs, axis=2)

        #    dimensionality: (connectedPairOfCells)
        dot_prods = np.sum(vec_diffs[:, 0] * vec_diffs[:, 1], axis=1)

        #    dimensionality: (connectedPairOfCells)
        angles = np.arccos(dot_prods / (mags[:, 0] * mags[:, 1]))

        have_ok_angle = angles <= max_angle
        have_ok_angle_indices = np.argwhere(have_ok_angle)[:, 0]

        return have_ok_angle_indices
