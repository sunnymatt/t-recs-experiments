import numpy as np
from trecs.models import ContentFiltering

# random recommender - randomly user-item scores at every step
class RandomRecommender(ContentFiltering):
    def _update_user_profiles(self, interactions):
        # do not change users_hat!
        self.predicted_scores[:, :] = np.random.uniform(size=(self.num_users, self.num_items))