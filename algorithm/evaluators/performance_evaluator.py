from abc import abstractmethod


class PerformanceEvaluator:
    def evaluate(self):
        """
        Evaluates performance
        """
        self.count_true()
        self.count_rec()
        self.count_true_rec()
        self.calc_metrics()

    @property
    def performance(self):
        if not (hasattr(self, "efficiency") or hasattr(self, "purity")):
            print('Call "evaluate" first to obtain the performance.')
            return {}

        return {"efficiency": self.efficiency, "purity": self.purity}

    def calc_metrics(self):
        """
        Calculates purity and efficiency

        - efficiency: proportion of the true cells that have been reconstructed 
        - purity: proportion of the reconstructed cells that are true cells
        """
        hascounts = (
            hasattr(self, "true_count")
            and hasattr(self, "rec_count")
            and hasattr(self, "true_rec_count")
        )

        if not hascounts:
            print("Counts must be calculated first.")
            return

        self.efficiency = self.true_rec_count / self.true_count
        self.purity = self.true_rec_count / self.rec_count

    @abstractmethod
    def count_true(self):
        pass

    @abstractmethod
    def count_rec(self):
        pass

    @abstractmethod
    def count_true_rec(self):
        pass

