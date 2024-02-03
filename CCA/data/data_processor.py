import numpy as np
import pandas as pd

from .event import Event
from .truth import Truth
from .hits import Hits


class DataProcessor:
    """
    Class that takes care of processing the data to get it ready for CA_track_finder
    """

    default_path_to_base = "/Users/archibaldruban/Google Drive/1. Education/UCL/1. MSci Theoretical Physics/_year 4/Master's Project/datasets/train_100_events"

    def __init__(self, path_to_base=None):
        self.raw_data = {}

        if path_to_base == None:
            self.path_to_base = self.default_path_to_base
        else:
            self.path_to_base = path_to_base

    def load(self, event_ids, content_to_load="all"):
        """
        Loads the events provided into self.raw_data
        
        content_to_load options: "all", cells", "hits", "particles", "truth"
        """
        for event_id in event_ids:
            # Loading content for event
            event = Event(event_id, self.path_to_base)
            event.load_content(content_to_load)

            # Saving event
            self.raw_data[event_id] = event

    def delete(self, event_ids):
        """
        Removes the events provided in self.raw_data
        
        event_ids: "all" | Array<event_id>
        """
        if event_ids == "all":
            self.raw_data = {}
        else:
            for event_id in event_ids:
                del self.raw_data[event_id]

    def process(
        self,
        event_id,
        pT_min=False,
        n_particles="all",
        remove_layer_duplicates=True,
        volume_ids=[8, 13, 17],
    ):
        """
        Returns a tuple (Hits, Truth)
        
        Processes the dataframe by:
        - Randomly selecting "n_particles" amount of particles
        - Filtering out hits from unwanted volumes,
        - Renaming layers as to make the volumes provided look like one big volume
        
        Parameters:
        - event_id: str
        - n_particles: "all" | int
        - pT_min: False | float - if float, applies cut by removing particles with pT < pT_min
        - volume_ids: array of volume_ids. Designed to work on adjacent barrel volumes 8, 13, 17
        but could work on similarly simply connected volumes.
        """
        event = self.raw_data[event_id]
        df_particles = event.data["particles"]
        df_truth = event.data["truth"]
        df_hits = event.data["hits"]

        if str(pT_min) != "False":
            df_particles = self.apply_pT_cut(df_particles, pT_min)

        # Randomly removing particles
        if type(n_particles) == int:
            df_particles = self.filter_randomly(df_particles, n_particles)

        # Collecting hits of corresponding particles in "truth"
        keep_indices = df_truth["particle_id"].isin(df_particles["particle_id"])
        df_truth = df_truth[keep_indices]

        # Collecting corresponding hits in "hits"
        keep_indices = df_hits["hit_id"].isin(df_truth["hit_id"])
        df_hits = df_hits[keep_indices]

        # Just a sanity / safety check that all the hits in the dataframes hits and truth are the same
        # It is to make sure that I am not filtering out any hits in truth when attempting to keep only
        # those in the relevant detector volumes (in the next line of code) (as df_truth doesn't have
        # any "volume_id" property)
        all_truth_in_hits = (
            df_truth[~df_truth["hit_id"].isin(df_hits["hit_id"])].shape[0] == 0
        )
        all_hits_in_truth = (
            df_hits[~df_hits["hit_id"].isin(df_truth["hit_id"])].shape[0] == 0
        )
        assert all_truth_in_hits and all_hits_in_truth

        # Keeping only wanted detector volumes
        df_hits = df_hits[df_hits["volume_id"].isin(volume_ids)]
        df_truth = df_truth[df_truth["hit_id"].isin(df_hits["hit_id"])]

        # Transform the layer_ids to 100*volume_id + layer_id
        for index, row in df_hits.iterrows():
            df_hits.at[index, "layer_id"] = self.processed_layer_id(
                row["volume_id"], row["layer_id"]
            )

        # REMOVING LAYER DUPLICATES MUST COME AFTER THE LAYER ID RENAMING
        if remove_layer_duplicates:
            df_hits, df_truth = self.remove_layer_duplicates(df_hits, df_truth)

        return Hits(df_hits), Truth(df_truth)

    def processed_layer_id(self, volume_id, layer_id):
        return int(100 * volume_id + layer_id)

    def apply_pT_cut(self, df_particles, pT_min):
        """
        Applies a cut on the transverse momentum pT of the particles, 
        where pT = sqrt(px^2 + py^2)
        
        returns df_particles
        """
        p_xy = df_particles[["px", "py"]].to_numpy()
        p_T = np.hypot(p_xy[:, 0], p_xy[:, 1])

        good_indices = np.argwhere(p_T > pT_min)[:, 0]

        good_particles = df_particles.iloc[good_indices]

        return good_particles

    def remove_layer_duplicates(self, df_hits, df_truth):
        """
        Removes duplicates of hits from the same particle that lie on the same layer
        
        returns df_hits, df_truth
        """
        df_merged = pd.merge(
            df_truth[["hit_id", "particle_id"]],
            df_hits[["layer_id"]],
            left_index=True,
            right_index=True,
        )

        particle_ids = df_merged["particle_id"].unique()
        layer_ids = df_merged["layer_id"].unique()

        drop_indices = []

        for particle_id in particle_ids:
            particle_hits = df_merged[df_merged["particle_id"] == particle_id]

            for layer_id in layer_ids:
                particle_layer_hits = particle_hits[
                    particle_hits["layer_id"] == layer_id
                ]

                if particle_layer_hits.shape[0] > 1:
                    drop_indices.extend(particle_layer_hits[1:].index)

        df_hits_filtered = df_hits.drop(drop_indices)
        df_truth_filtered = df_truth.drop(drop_indices)

        return df_hits_filtered, df_truth_filtered

    def filter_randomly(self, df, keep_n):
        """
        Returns a subset of df (Numpy array or Pandas dataframe) with 
        'keep_n' of its rows selected randomly
        """
        remove_n = max(df.shape[0] - keep_n, 0)

        # indices to remove
        drop_indices = np.random.choice(df.index, remove_n, replace=False)

        # subset
        subset = df.drop(drop_indices)

        return subset

