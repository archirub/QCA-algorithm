from functools import reduce
import numpy as np


class Tracks(dict):
    """
    Holds tracks as a dictionary of shape:
    {
        [track_length]: Numpy array of cell_ids
    }
    
    properties:
    - changed: 
    """

    def __init__(self, dict_, cells):
        super().__init__(dict_)
        self.cells = cells

    @property
    def size(self):
        """
        Gives the total number of tracks 
        """
        return reduce(lambda tot, arr: tot + arr.shape[0], self.values(), 0)

    def to_hits(self, sort_hits=True):
        """
        returns tracks as a dict of shape
        {
            [track length]: Numpy array of hit_ids
        }
        """
        attr_name = "_sorted_hits" if sort_hits else "_hits"

        if hasattr(self, attr_name):
            return getattr(self, attr_name)

        tracks_as_hits = {}

        for track_length, tracks in self.items():
            # tracks as hits are 1 longer than tracks as cells
            new_track_length = track_length + 1

            if len(tracks) < 1:
                continue

            most_hits = self.cells[tracks][:, :, 1]
            last_hit = self.cells[tracks][:, -1, 0]

            track_as_hits = np.hstack((most_hits, last_hit[..., np.newaxis]))

            tracks_as_hits[new_track_length] = (
                np.sort(track_as_hits) if sort_hits else track_as_hits
            )

        setattr(self, attr_name, tracks_as_hits)

        return getattr(self, attr_name)

    def to_particles(self):
        pass


# def hits_to_particles(tracks_as_hits, map_hit_to_particle):
#     return {
#         track_length: np.array([
#             [map_hit_to_particle[hit_id] for hit_id in track] for track in tracks
#         ]) for track_length, tracks in tracks_as_hits.items()
#     }
