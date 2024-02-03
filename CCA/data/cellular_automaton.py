import numpy as np

class CellularAutomaton(dict):
    """
    Class specifically for this tracking algorithm (not a general CA class)
    
    Inherits from dict, such that self gives the CA.
    
    The CA is a dict of shape
    {
        [cell_id]: {
            "state": positive int
            "inner_neighs": array of cell_ids
        }
    }
    """    
    def __init__(self, CA):
        super().__init__(CA)
    
    @property
    def states(self):
        """
        Returns a Numpy array of all the states of the CA
        """
        return np.array([val["state"] for val in self.values()])
    
    
    def states_dict(self):
        """
        Returns a dictionary of shape {cell_id: state}
        """
        return {cell_id: cell["state"] for cell_id, cell in self.items()}
    
    def remove_cells(self, r_cells):
        """
        Removes cells by:
        - deleting their entry (i.e. removing the vertices)
        - Removing them from all inner_neighs arrays (i.e. removing the edges)

        r_cells: array of cells to remove
        """
        # creating a dict for speed
        r_cells_dict = {cell_id: None for cell_id in r_cells}
        
        # Removing cells' entry
        for cell_index in r_cells_dict.keys():
            del self[cell_index]

        # Removing cells in inner_neighs
        for cell in self.values():
            neighs = cell["inner_neighs"]

            # Looping backwards to avoid change in indexing
            for i in reversed(range(len(neighs))):
                neigh_index = neighs[i]

                if neigh_index in r_cells_dict:
                    del neighs[i]
        