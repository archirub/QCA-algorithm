import pandas as pd
import numpy as np
from CCA.data.custom_tools import ignore_warning


class Truth(pd.DataFrame):
    """
    Extension of the hits dataframe from the dataset
    with all pandas.DataFrame properties and additionally:
    - hit_ids
    - particle_ids
    - to_dict()
    """

    def __init__(self, *args, **kwargs):
        super(Truth, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Truth

    @property
    @ignore_warning(UserWarning)
    def hit_ids(self):
        """
        Numpy array of hits ids (w/ duplicates)
        """
        if not hasattr(self, "_hids"):
            self._hids = self["hit_id"].to_numpy()

        return self._hids

    @property
    @ignore_warning(UserWarning)
    def particle_ids(self):
        """
        Numpy array of particle ids (w/ duplicates)
        """
        if not hasattr(self, "_pids"):
            self._pids = self["particle_id"].to_numpy()

        return self._pids

    def to_track_dict(self):
        """
        Returns a dictionary of shape
            {
                track_length: Array of tracks 
            }

        each track is an array of *sorted* hit_ids
        """
        if hasattr(self, "_dict"):
            return self._dict

        particle_ids = self.particle_ids
        unique_particle_ids = np.unique(self.particle_ids)
        hits_ids = self.hit_ids

        true_tracks_dict = {}

        for particle_id in unique_particle_ids:
            indices = np.argwhere(particle_ids == particle_id)[:, 0]
            track = np.sort(hits_ids[indices])
            track_length = len(track)

            if track_length in true_tracks_dict:
                true_tracks_dict[track_length] = np.vstack(
                    (true_tracks_dict[track_length], track)
                )
            else:
                true_tracks_dict[track_length] = np.array([track])

        self._dict = true_tracks_dict

        return self._dict

