import numpy as np
from trecs.metrics import Measurement

# measure # of items shown to users that are created by members of Group A
class MajorityRecommendationShare(Measurement):
    def __init__(self, attr_index, name="majority_dominance", verbose=False):
        self.attr_index = attr_index # which item profile tells us which group it came from?
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        # measure average percentage of items that are created by Group A
        items_shown = kwargs.pop("items_shown", None)
        item_vector = items_shown.reshape(-1)
        # in popularity recommender, users are shown the same items
        perc_majority = sum(recommender.items[self.attr_index, item_vector]) / len(item_vector)
        self.observe(perc_majority)

# look at Group A's dominance of the top X% of popular items
class MajoritySharePopularItems(Measurement):
    def __init__(self, total_items, attr_index, top_percentage=0.05, name="majority_share_most_popular", verbose=False):
        self.num_most_popular = round(top_percentage * total_items)
        self.attr_index = attr_index
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        # the following item takes recommender.items_hat, which is a 1 x num_items array containing
        # the popularity of item i at index i, where popularity is defined as # of interactions from users,
        # and returns the indices of the most popular items.
        # we then calculate the percentage of most popular items that were created by the majority group
        top_ids = np.argpartition(recommender.items_hat.reshape(-1), -self.num_most_popular)[-self.num_most_popular:]
        perc_majority = sum(recommender.items[self.attr_index, top_ids]) / len(top_ids)
        self.observe(perc_majority)

# creators' group-sensitive attribute
class CreatorAvgGroupSkew(Measurement):
    def __init__(self, name="creator_group_skew", verbose=False):
        Measurement.__init__(self, name, verbose, init_value=None)

    def measure(self, recommender, **kwargs):
        group_attr = recommender.creators.actual_creator_profiles[:, -1]
        self.observe(group_attr)