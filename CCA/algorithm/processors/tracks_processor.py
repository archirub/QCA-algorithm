import copy
import numpy as np

from CCA.data.tracks import Tracks


class TracksProcessor:
    def __init__(self, evolutionProcessor, min_length):
        """
        - min_length: minimum track length generated
        """
        self.CA = evolutionProcessor.CA
        self.cells = evolutionProcessor.cells
        self.min_length = min_length

    #     @property
    #     def tracks_as_hits(self)
    #         if not hasattr(self, "tracks"):
    #             print("Must generate tracks first.")

    #         return tracks.to_hits(self.cells)

    def generate(self, final_print=True, progress_print=False):
        """
        Generates an array of all track candidates from CA

        returns a dictionary of shape:
        {
            [track_length]: Numpy array of cell_ids
        }

        This shape was chosen so that each value of the dict can be a Numpy array for fast processing
        """
        print("Generating track candidates...")

        # Copying for modifying without modyfing orignal CA
        CA = copy.deepcopy(self.CA)

        max_length = max(CA.states)
        track_lengths = np.arange(self.min_length, max_length + 1)

        # Looping through track lengths starting from longest
        all_track_candidates = {}
        for track_length in reversed(track_lengths):
            track_candidates = []

            if progress_print:
                print(f"Processing tracks of length {track_length}...", end="\r")

            all_cell_indices = np.array(list(CA.keys()))
            all_states = np.array([CA[i]["state"] for i in all_cell_indices])

            indices = np.argwhere(all_states == track_length)[:, 0]
            state_cell_indices = all_cell_indices[indices]

            # Looping through cells of state = track_length
            for j in range(len(state_cell_indices)):
                cell_index = state_cell_indices[j]
                cell = CA[cell_index]

                # Tree recursion through inner neighbours
                self.tree_recursion_neighs(cell_index, CA, [], track_candidates)

            # Removing side-product candidates smaller than track_length
            for i in reversed(range(len(track_candidates))):
                if len(track_candidates[i]) < track_length:
                    track_candidates.pop(i)

            track_candidates = np.array(track_candidates)
            all_track_candidates[track_length] = track_candidates

            # Remove cells in track_candidates from CA graph
            CA.remove_cells(np.unique(track_candidates.flatten()))

        self.tracks = Tracks(all_track_candidates, self.cells)

        if final_print:
            print(f"{self.tracks.size} tracks generated.")

    def tree_recursion_neighs(self, cell_index, CA, chain, chains):
        """
        Recursive function to generate the track candidates for a given (largest state) cell as the seed. 

        The stop condition of my recursion is when the current cell doesn't have any more inner neighbours. 

        Into the recursion function should be passed the current cell, its id, and an array of its parent cell indices, that is, the previous elements in the track candidate constructed so far.

        If the recursion needs to stop, then the current cell_index is just added to the chain and the chain is returned, 
        If it needs to continue, then we loop over the inner neighbours and call the recursive function 
        for each inner neighbour, and append the result of the recursion to the track candidate.
        """
        cell = CA[
            cell_index
        ]  # not using self.CA because CA in self.generate() is not the same as self.CA
        chain.append(cell_index)

        if len(cell["inner_neighs"]) < 1:
            chains.append(chain)

        else:
            for neigh_index in cell["inner_neighs"]:
                new_chain = chain.copy()
                self.tree_recursion_neighs(neigh_index, CA, new_chain, chains)

