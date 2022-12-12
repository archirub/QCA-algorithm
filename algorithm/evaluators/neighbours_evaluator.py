import numpy as np
from .performance_evaluator import PerformanceEvaluator


class NeighboursEvaluator(PerformanceEvaluator):
    def __init__(self, truth, neighboursProcessor):
        self.truth = truth
        self.CA = neighboursProcessor.CA
        self.cells = neighboursProcessor.cells

    def count_true(self):
        """
        Returns the total number of true neighbours in the sample "df_truth"

        Takes the # of true neighbours to be the # of neighbours across all true tracks.
        In a given true track, the # of neighbours is calculated as 
        """
        particle_ids = self.truth.particle_ids

        # Extracting the count for each pairs
        unique, hit_counts = np.unique(particle_ids, return_counts=True)

        neighs_counts = hit_counts - 2

        # filtering out -ve counts (those made of just 1 hit)
        neighs_counts[neighs_counts < 0] = 0

        total_neigh_count = np.sum(neighs_counts)

        self.true_count = total_neigh_count

    def count_rec(self):
        """
        Returns the total number of neighbours that were reconstructed.

        This is equal to the total # of inner neighbours
        """
        count = 0

        for cell in self.CA.keys():
            count += len(self.CA[cell]["inner_neighs"])

        self.rec_count = count

    def count_true_rec(self):
        """
        Returns the total number of true pairs that were reconstructed
        """
        count = 0

        # Generates a dict containing all hits and of shape {hit_id: particle_id}
        map_hit_to_particle = {
            item[0]: item[1]
            for item in self.truth[["hit_id", "particle_id"]].to_numpy()
        }

        # Loops through all cells
        for cell_index, cell_info in self.CA.items():
            hit_id3 = self.cells[cell_index][1]
            hit_id2 = self.cells[cell_index][0]
            particle_id3 = map_hit_to_particle[hit_id3]
            particle_id2 = map_hit_to_particle[hit_id2]

            # Checks whether the particle_id of the two hits of the cell match
            if particle_id3 == particle_id2:
                # Loops through the inner neighbours
                for neigh_index in cell_info["inner_neighs"]:
                    hit_id1 = self.cells[neigh_index][0]  # inner hit of inner neigh
                    particle_id1 = map_hit_to_particle[hit_id1]

                    # Check whether the inner hit of the inner neighbour also has the same particle id,
                    # in which case all 3 hits of the neighbour bound are from the same true particle
                    if particle_id1 == particle_id2:
                        count += 1

        self.true_rec_count = count

