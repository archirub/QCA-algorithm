from CCA.data import CellularAutomaton


class EvolutionProcessor:
    def __init__(self, neighboursProcessor):
        self.CA = neighboursProcessor.CA
        self.cells = neighboursProcessor.cells

    def evolve(self, final_print=True, progress_print=False):
        """
        Runs the evolution of the cellular automaton on the cells provided.

        Takes in and returns a dictionary of shape:
        {
         ["cell_index1"]: {
             "inner_neighs": [arr of cell_indices],
             "state": integer,
         },
         ["cell_index2"]: {...},
         ...
        }

        """
        print("Evolving CA...")

        # Creating state dictionaries for faster dict copying (necessitated after each iteration)
        states_curr = self.CA.states_dict()
        states_next = states_curr.copy()

        cell_indices = self.CA.keys()

        iters = 0
        state_changed = True

        # While the state of at least one cell has changed
        while state_changed:
            iters += 1
            state_changed = False

            if progress_print:
                print("Iteration", iters)

            # POSSIBLE OPTIMIZATION: could only loop starting at 2nd layer since 1st layer has no inner neighs

            # Looping through cells
            for cell_index in cell_indices:
                cell_state = states_curr[cell_index]
                neighs = self.CA[cell_index]["inner_neighs"]

                # Looping through inner neighbours of cells
                for neigh_index in neighs:
                    neigh_state = states_curr[neigh_index]

                    # Increment the state of the cell if one of its inner neighbours
                    # has the same state as it
                    if cell_state == neigh_state:
                        states_next[cell_index] = states_curr[cell_index] + 1
                        state_changed = True
                        break

            states_curr = states_next.copy()

        # Creating final object
        CA_final = {
            cell_id: {
                "state": states_next[cell_id],
                "inner_neighs": cell["inner_neighs"],
            }
            for cell_id, cell in self.CA.items()
        }
        self.CA = CellularAutomaton(CA_final)

        if final_print:
            print("Evolution completed.")
            print("")

