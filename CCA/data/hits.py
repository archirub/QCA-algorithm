import pandas as pd

from CCA.data.custom_tools import ignore_warning


class Hits(pd.DataFrame):
    """
    Extension of the hits dataframe from the dataset with all pandas.DataFrame properties and additionally:
    - layer_ids()
    - hit_ids()
    """

    def __init__(self, *args, **kwargs):
        super(Hits, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Hits

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
    def layer_ids(self):
        """
        Numpy array of layer ids (w/o duplicates)
        """
        if not hasattr(self, "_lids"):
            self._lids = self["layer_id"].drop_duplicates().to_list()

        return self._lids

