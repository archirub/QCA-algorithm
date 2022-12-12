import numpy as np
from .performance_evaluator import PerformanceEvaluator


class TracksEvaluator(PerformanceEvaluator):
    def __init__(self, truth, tracksProcessor):
        self.truth = truth
        self.min_length = tracksProcessor.min_length
        self.tracks = tracksProcessor.tracks

    #         self.CA = neighboursProcessor.CA
    #         self.cells = neighboursProcessor.cells

    def count_true(self):
        """
        Returns the total number of true tracks in the sample "df_truth"

        # of true tracks is taken to be the # of unique particle_ids that have
        at least "min_track_length" hits
        """
        particle_id_counts = self.truth["particle_id"].value_counts()
        particle_id_counts = particle_id_counts[
            particle_id_counts.values >= self.min_length
        ]

        self.true_count = particle_id_counts.shape[0]

    def count_rec(self):
        """
        Returns the total number of reconstructed tracks.

        """
        self.rec_count = self.tracks.size

    def count_true_rec(self):
        """    
        Returns the number of true tracks that are exactly reconstructed by a candidat track

        candidate_dict and true_dict both have shape:
            {
                track_length: Array of tracks
            }

        where each track is itself an array of its hit_ids, *ordered*.
        """
        candidate_dict = self.tracks.to_hits()
        true_dict = self.truth.to_track_dict()

        count = 0

        for track_length, true_tracks in true_dict.items():
            if not track_length in candidate_dict:
                continue

            candidate_tracks = candidate_dict[track_length]

            for true_track in true_tracks:
                is_reconstructed = np.any(
                    np.all(true_track == candidate_tracks, axis=1)
                )

                if is_reconstructed:
                    count += 1

        self.true_rec_count = count

