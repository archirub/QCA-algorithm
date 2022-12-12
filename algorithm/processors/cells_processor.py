import numpy as np


class CellsProcessor:
    """
    Processor for the cells of the classical CA
    
    Properties:
    - cells: (holds cells) 2D array which consists of an array of arrays of length 2 which contain
        the inner and outer "hit_id"'s for each cell (shape is (cell, hit_id))
    - cells_per_layer: a dictionary used in neighbour finding. It has shape:
        {
            ["layer_id"]: [
                        index of 1st cell of that layer in cells array, 
                        index of last cell of that layer in cells array
                        ]
        }
    - cells_positions: has dimensionality (cell, hit, hit_coordinates). The cells and the hits
    
    """

    def __init__(self, hits, min_angle):
        """
        hits: Hits class instance
        min_angle: (float) minimum cone angle allowed between two hits to form a pair
        """
        self.hits = hits
        self.min_angle = min_angle

        self.cells = np.array([])
        self.cells_per_layer = {}
        self.cells_positions = np.array([])

    def __str__(self):
        return str(self.cells)

    def __repr__(self):
        return repr(self.cells)

    def form(self, final_print=True, progress_print=False):
        """
        (20x faster than "form_pairs" in part 1)

        Forms all possible cells as pairs of hits from the hits provided.

        Two hits form a pair if:
        - They lie on adjacent layers
        - The outer hit lies within a cone of angle self.min_angle relative to the inner hit

        returns (cells, cells_per_layer) where:
        - cells is 2D array which consists of an array of arrays of length 2 which contain
        the inner and outer "hit_id"'s for each cell (shape is (cell, hit_id))
        - cells_per_layer is a dictionary used in neighbour finding. It has shape:
            {
                ["layer_id"]: [
                            index of 1st cell of that layer in cells array, 
                            index of last cell of that layer in cells array
                            ]
            }
        - cells_positions: has dimensionality (cell, hit, hit_coordinates). The cells and the hits
        are in the same order as in the array "cells"
        """
        print("Forming pairs...")

        layer_ids = self.hits["layer_id"].drop_duplicates().to_list()
        layer_ids.sort()

        cells = []
        cells_positions = []

        # Creating dictionary to map layers to their cells
        # The format will be {"layer_id": [index of 1st cell of that layer in cells array, index of last cell of that layer in cells array]}
        cells_per_layer = {}

        # Looping through layer pairs
        for i in range(len(layer_ids) - 1):
            if progress_print:
                print(
                    "Processing {0}/{1} layer pairs.".format(i + 1, len(layer_ids) - 1)
                )
            inner_layer_id = layer_ids[i]
            outer_layer_id = layer_ids[i + 1]

            # Loading hits for each layer of the layer pair
            inner_hits = self.hits[self.hits["layer_id"] == inner_layer_id][
                ["hit_id", "x", "y", "z"]
            ].to_numpy()
            outer_hits = self.hits[self.hits["layer_id"] == outer_layer_id][
                ["hit_id", "x", "y", "z"]
            ].to_numpy()

            # For "cells_per_layer"
            cells_n_before = len(cells)

            # Mesh grid of positions (manually done because couldn't np.meshgrid to work for 2D arrays)
            inner_hits_pos = inner_hits[:, 1:]
            outer_hits_pos = outer_hits[:, 1:]
            inner_hits_pos_mesh = np.tile(
                inner_hits_pos[:, np.newaxis, :], (1, outer_hits.shape[0], 1)
            )
            outer_hits_pos_mesh = np.tile(
                outer_hits_pos[np.newaxis, :, :], (inner_hits.shape[0], 1, 1)
            )

            # Calculating whether in the inner hit's cone in a batch
            are_good = self.is_in_cone(
                inner_hits_pos_mesh, outer_hits_pos_mesh, inner_hits_pos
            )

            # Getting the corresponding cells
            good_hit_indices = np.argwhere(are_good)

            if good_hit_indices.shape[0] > 0:
                good_inner_hits = inner_hits[good_hit_indices[:, 0]]
                good_outer_hits = outer_hits[good_hit_indices[:, 1]]

                # forming pairs
                pairs = np.array([good_inner_hits, good_outer_hits])
                pairs = np.transpose(pairs, axes=(1, 0, 2))

                # Separating hit ids and hit positions
                pairs_ids = pairs[:, :, 0]
                pairs_pos = pairs[:, :, 1:]

                # Appending to respective arrays
                cells.extend(pairs_ids)
                cells_positions.extend(pairs_pos)

            # Storing the location (in the array cells) of the cells
            # whose 1st hit is in this particular layer
            cells_per_layer[inner_layer_id] = [cells_n_before, len(cells)]

        cells = np.array(cells, dtype=int)
        cells_positions = np.array(cells_positions)

        if final_print:
            print("{0} cells formed.".format(cells.shape[0]))
            print("")

        self.cells = cells
        self.cells_per_layer = cells_per_layer
        self.cells_positions = cells_positions

    def is_in_cone(self, inner_hits_pos_mesh, outer_hits_pos_mesh, inner_hits_pos):
        """
        (adapted from part 1 to work on mesh (for faster form_pairs))

        Returns a Boolean indicating whether the outer hit lies within a cone
        with tip located at the inner hit's position and whose main axis
        points in the direction from the origin (i.e. the collision) to the inner hit

        Takes arrays of positions for inner_hit and outer_hit with dimensions (hit_count, 3)

        alpha is the inner angle between the cone's main axis and its surface.
        0 < alpha < pi/2
        """
        assert self.min_angle < np.pi / 2

        r_21 = outer_hits_pos_mesh - inner_hits_pos_mesh

        norm_inner = np.linalg.norm(inner_hits_pos, axis=1)[..., np.newaxis]
        unit_r_1 = inner_hits_pos / norm_inner
        cone_dist = np.sum(r_21 * unit_r_1[:, np.newaxis], axis=-1)

        # Finding radius of cone at the outer hit's projection onto the main axis
        cone_radius = cone_dist * np.tan(self.min_angle)

        # Finding orthogonal distance of outer hit from main axis
        orth_distance = np.linalg.norm(
            r_21 - cone_dist[..., np.newaxis] * unit_r_1[:, np.newaxis], axis=-1
        )

        # If dot product yields negative, then the outer hit is on the other side of the collision
        are_on_right_side = cone_dist > 0
        are_within_cone = orth_distance < cone_radius

        return np.logical_and(are_on_right_side, are_within_cone)

