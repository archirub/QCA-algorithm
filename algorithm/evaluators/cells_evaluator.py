import numpy as np
from .performance_evaluator import PerformanceEvaluator


class CellsEvaluator(PerformanceEvaluator):
    def __init__(self, truth, cellsProcessor):
        self.truth = truth
        self.cells = cellsProcessor.cells
        

    def count_true(self):
        """
        Calculates the total number of true pairs / cells in the sample "df_truth"
        """
        particle_ids = self.truth.particle_ids

        # Extracting the count for each pairs
        unique, hit_counts = np.unique(particle_ids, return_counts=True)

        pair_counts = hit_counts - 1

        total_pair_count = np.sum(pair_counts)

        self.true_count = total_pair_count


    def count_rec(self):
        """
        Calculates the total number of pairs that were reconstructed (simply the length of cells)
        """
        self.rec_count = len(self.cells)


    def count_true_rec(self):
        """
        Calculates the total number of true pairs that were reconstructed
        """
        count = 0

        hit_ids = self.truth.hit_ids
        particle_ids =  self.truth.particle_ids

        for cell in self.cells:
            hit1_index = np.where(hit_ids == cell[0])[0][0]
            hit2_index = np.where(hit_ids == cell[1])[0][0]

            hit1_particle_id = particle_ids[hit1_index]
            hit2_particle_id = particle_ids[hit2_index]

            if hit1_particle_id == hit2_particle_id:
                count += 1
    
        self.true_rec_count = count
        
        
