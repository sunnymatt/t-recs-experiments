import numpy as np
from trecs.metrics import Measurement


class GroupMSEMeasurement(Measurement):
    def __init__(self, attr_index, name="group_mse", verbose=False):
        self.attr_index = attr_index
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        maj_indices = np.where(
            recommender.users.actual_user_profiles[:, self.attr_index] > 0
        )
        min_indices = np.where(
            recommender.users.actual_user_profiles[:, self.attr_index] == 0
        )
        maj_mse = (
            (
                recommender.predicted_scores[maj_indices]
                - recommender.users.actual_user_scores[maj_indices]
            )
            ** 2
        ).mean()
        min_mse = (
            (
                recommender.predicted_scores[min_indices]
                - recommender.users.actual_user_scores[min_indices]
            )
            ** 2
        ).mean()

        self.observe([maj_mse, min_mse], copy=False)