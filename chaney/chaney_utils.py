"""
Additional classes / methods to support replication of Chaney et al.
"""
import numpy as np
from collections import defaultdict
import os
import pickle as pkl
from trecs.metrics import Measurement
from trecs.models import ContentFiltering
from trecs.matrix_ops import normalize_matrix, inner_product
from trecs.random import Generator
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def gen_social_network(user_prefs):
    """ Generates a |U|x|U| social network of connections
        as specified in Chaney et al.
    """
    user_cov = np.cov(user_prefs)
    possible_thresholds = np.sort(user_cov.flatten())[::-1]
    user_connections = None
    num_users = user_prefs.shape[0]
    for thresh in possible_thresholds[num_users:]:
        num_connected = (user_cov >= thresh).any(axis=1).sum()
        if num_connected == num_users:
            return (user_cov >= thresh).astype(int) # final adjacency matrix
    raise RuntimeError("Could not find a suitable threshold.")


def mu_sigma_to_alpha_beta(mu, sigma):
    """ For Chaney's custom Beta' function, we convert
        a mean and variance to an alpha and beta parameter
        of a Beta function. See footnote 3 page 3 of Chaney
        et al. for details.
    """
    alpha = ((1-mu) / (sigma**2) - (1/mu)) * mu**2
    beta = alpha * (1/mu - 1)
    return alpha, beta

def exclude_new_items(num_items_per_iter):
    """ This custom scoring function ensures that all items in the system
        that are created after a certain point are given a score of negative
        infinity, ensuring that they will be at the very bottom of any recommendation
        list. This ensures that only the items in the training phase get recommended;
        new items are only recommended via the interleaving procedure.
    """
    # score_fn is called by process_new_items and by train.
    # therefore, when score_fn is being called the first time, we will give all new items
    # scores of negative infinity; then, when train() is called, the actual
    # scores will be supplied.
    def score_fn(users, items):
        predicted_scores = inner_product(users, items)
        if items.shape[1] == num_items_per_iter: # EDGE CASE: when num_items_per_iter = num_items
            # all predicted scores for these "new" items will be negative infinity,
            # ensuring they never get recommended
            predicted_scores[:, :] = float('-inf')
        return predicted_scores
    return score_fn

def perfect_scores(num_items_per_iter, true_scores):
    """ This custom scoring function ensures that all items in the system
        that are created after a certain point are given a score of negative
        infinity, ensuring that they will be at the very bottom of any recommendation
        list. Otherwise, we return the "true scores" specified in the
        true_scores array. Additionally, it ensures that the true utilities are
        returned for each item. This is the scoring used by the IdealRecommender.
    """
    score_copy = np.copy(true_scores)
    def score_fn(users, items):
        predicted_scores = np.copy(score_copy)
        num_users, num_items = users.shape[0], items.shape[1]
        # all predicted scores for these "new" items will be negative infinity,
        # ensuring they never get recommended; instead, they are interleaved into the recommendation
        # set
        predicted_scores = predicted_scores[:num_users, :num_items] # subset to correct dimensions
        if items.shape[1] == num_items_per_iter: # EDGE CASE: when num_items_per_iter = num_items
            # all predicted scores for these "new" items will be negative infinity,
            # ensuring they never get recommended
            predicted_scores[:, :] = float('-inf')
        return predicted_scores
    return score_fn

def interleave_new_items(generator):
    """ Chooses the most recent, newest items to interleave
        with the recommendation set. This custom interleaving method
        ensures that all of the most recently created items (i.e.,
        the newest items) are the ones interleaved with the recommendations.
    """
    def interleaving_fn(k, item_indices):
        num_users = item_indices.shape[0]
        indices = item_indices[:, -k:]
        values = generator.random(indices.shape)
        order = values.argsort(axis=1) # randomly sort indices within rows
        rows = np.tile(np.arange(num_users).reshape((-1, 1)), indices.shape[1])
        return indices[rows, order]
    return interleaving_fn

# utility function to extract measurement
def process_measurement(model, metric_string):
    # get rid of the None value at the beginning
    return model.get_measurements()[metric_string][1:]

def load_sim_results(folder, filename="sim_results.pkl"):
    filepath = os.path.join(folder, filename)
    return pkl.load(open(filepath, "rb"))

def merge_results(folder_paths):
    """
    Paths must be paths to pickle files resulting from multiple trials of the same
    simulation setup
    """
    final_result = defaultdict(lambda: defaultdict(list))

    for folder_path in folder_paths:
        results = load_sim_results(folder_path)
        # key = metric, value = dictionary mapping algo name to list of entries
        for metric_name, v in results.items():
            for model_name, metric_vals in v.items():
                # merge the list
                final_result[metric_name][model_name] += metric_vals

    return final_result

"""
Graphing results utilities
"""
def transform_relative_to_ideal(train_results, metric_key, model_keys, absolute_measure=True):
    relative_dist = defaultdict(lambda: defaultdict(list))

    if absolute_measure:
        ideal_dist = np.array(train_results[metric_key]["ideal"])
    else:
        model_key = list(train_results[metric_key].keys())[0]
        # zeros for all timsteps
        trials = len(train_results[metric_key][model_key])
        timesteps = len(train_results[metric_key][model_key][0])
        ideal_dist = np.zeros((trials, timesteps))
        relative_dist[metric_key]["ideal"] = ideal_dist

    for model_key in model_keys:
        if model_key is "ideal" and not absolute_measure:
            continue

        abs_dist = np.array(train_results[metric_key][model_key])
        if absolute_measure:
            abs_dist = abs_dist - ideal_dist
        relative_dist[metric_key][model_key] = abs_dist
    return relative_dist

def graph_metrics(train_results, metric_key, model_keys, label_map, mean_sigma=0, mult_sd=0, conf_sigma=0):
    for m in model_keys:
        if not isinstance(train_results[metric_key][m], np.ndarray):
            train_results[metric_key][m] = np.array(train_results[metric_key][m])
        # average across trials and smooth, if necessary
        if mean_sigma > 0:
            values = gaussian_filter1d(train_results[metric_key][m].mean(axis=0), sigma=mean_sigma)
        else:
            values = train_results[metric_key][m].mean(axis=0)
        line = plt.plot(values, label=label_map[m])
        line_color = line[0].get_color()
        if mult_sd > 0:
            std = train_results[metric_key][m].std(axis=0)
            timesteps = np.arange(len(std))
            low = values - mult_sd * std
            high = values + mult_sd * std
            if conf_sigma > 0:
                low = gaussian_filter1d(low, sigma=conf_sigma)
                high = gaussian_filter1d(high, sigma=conf_sigma)
            plt.fill_between(timesteps, low, high, color = line_color, alpha=0.3)
    plt.legend(facecolor='white', framealpha=1, loc='upper right', bbox_to_anchor=(1.7, 1.0))

def graph_relative_to_ideal(train_results, metric_key, model_keys, label_map, absolute_measure=True, mean_sigma=0, mult_sd=0, conf_sigma=0):
    relative_dist = transform_relative_to_ideal(train_results, metric_key, model_keys, absolute_measure)
    graph_metrics(relative_dist, metric_key, model_keys, label_map, mean_sigma, mult_sd, conf_sigma)

def last_timestep_values(train_results, metric_key, model_keys, absolute_measure=True):
    relative_dist = transform_relative_to_ideal(train_results, metric_key, model_keys, absolute_measure)
    for model_key in relative_dist.keys():
        # just take out last value
        relative_dist[model_key] = relative_dist[model_key][:, -1]
        sd = relative_dist[model_key].std()
        print(f"Mean value of {metric_key} for model {model_key}: {relative_dist[model_key].mean(axis=0):.2f} (sd: {sd:.3f})")

    return relative_dist


"""
Custom measurements
"""

def calculate_avg_jaccard(pairs, interactions):
    """ Calculates average Jaccard index over specified pairs of users.
    """
    similarity = 0
    num_pairs = len(pairs)
    for user1, user2 in pairs:
        itemset_1 = set(interactions[user1, :])
        itemset_2 = set(interactions[user2, :])
        common = len(itemset_1.intersection(itemset_2))
        union = len(itemset_1.union(itemset_2))
        similarity += common / union / num_pairs
    return similarity


class InteractionTracker(Measurement):
    """ Tracks all user interactions up to the current timepoint. In the context of replication,
        it will be used to track the items that users interact with in the ideal
        recommender. We need this in order to calculate homogenization of other non-ideal
        RS algorithms, relative to the ideal recommender.
    """
    def __init__(self, name="interaction_history", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        self.observe(np.copy(interactions).reshape((-1, 1)))


class SimilarUserInteractionSimilarity(Measurement):
    """
    Measures the homogenization of users deemed most similar by the RS algorithm,
    relative to the homogenization those users face under the ideal recommender.
    """
    def __init__(self, ideal_interaction_hist, name="similar_user_jaccard", seed=None, verbose=False):
        self.ideal_hist = ideal_interaction_hist
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.timestep += 1;
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        # generate cosine similarity matrix for all users
        assert recommender.users_hat.get_timesteps() == self.timestep + 1 # ensure that the users_hat variable is storing copies at each timestep
        user_representation = recommender.users_hat.state_history[-1]
        sim_matrix = cosine_similarity(user_representation, user_representation)
        # set diagonal entries to zero
        num_users = sim_matrix.shape[0]
        sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = sim_matrix
        sim_tiebreak["random"] = self.rng.random(sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        ideal_similarity = calculate_avg_jaccard(pairs, self.ideal_hist[:, :(self.timestep + 1)]) # compare
        this_similarity = calculate_avg_jaccard(pairs, self.interaction_hist)
        self.observe(this_similarity - ideal_similarity)
        self.timestep += 1 # increment timestep


# Calculate homogenization by the average Euclidean distance of the interaction set

def avg_interaction_distance(items1, items2, item_attributes):
    """
    Assumes items are provided in timestep order;
    averages the euclidean distance over timesteps.

    Assume items matrix is |A| x |I|
    """
    num_steps = len(items1)
    assert len(items1) == len(items2) # should have interacted with same # of itesm
    total_distance = 0
    for i in range(num_steps):
        item1 = item_attributes[:, items1[i]]
        item2 = item_attributes[:, items2[i]]
        total_distance += np.linalg.norm(item1 - item2)
    return total_distance / num_steps

def distance_of_mean_items(items1, items2, item_attributes):
    """
    Returns the difference between the average vector of the items
    in set 1 and the average vector of the items in set 2.

    Assume items matrix is |A| x |I|
    """
    mean1 = item_attributes[:, items1].mean(axis=1)
    mean2 = item_attributes[:, items2].mean(axis=1)
    return np.linalg.norm(mean1 - mean2)

def mean_item_dist_pairs(pairs, interaction_history, item_attributes):
    """
    For each pair, calculates the distance between the mean item
    interacted with by each member of the pair. Then averages these
    distances across all pairs.
    """
    dist = 0
    for pair in pairs:
        itemset_1 = interaction_history[pair[0], :].flatten()
        itemset_2 = interaction_history[pair[1], :].flatten()
        dist += distance_of_mean_items(itemset_1, itemset_2, item_attributes) / len(pairs)
    return dist

class MeanInteractionDistance(Measurement):
    """
    Cacluates the mean distance between items in each users' recommendation list based on their item attributes
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mean_rec_distance")
            Name of the measurement component.
    """
    def __init__(self, pairs, name="mean_interaction_dist", verbose=False):
        Measurement.__init__(self, name, verbose)
        self.pairs = pairs
        self.interaction_hist = None

    def measure(self, recommender):
        """
        TODO
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])

        avg_dist = mean_item_dist_pairs(self.pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(avg_dist)

class MeanDistanceSimUsers(Measurement):
    """
    Cacluates the mean distance between items in each users' interaction list based on their item attributes
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mean_rec_distance")
            Name of the measurement component.
    """
    def __init__(self, ideal_interaction_hist, ideal_item_attrs, seed=None, name="sim_user_dist", verbose=False):
        self.ideal_hist = ideal_interaction_hist
        self.ideal_item_attrs = ideal_item_attrs
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        TODO
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        interactions = recommender.interactions
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])

        # get value of user matrix
        user_representation = recommender.users_hat.state_history[-1]
        # find most similar users
        sim_matrix = cosine_similarity(user_representation, user_representation)
        # set diagonal entries to zero
        num_users = sim_matrix.shape[0]
        sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = sim_matrix
        sim_tiebreak["random"] = self.rng.random(sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        ideal_hist = self.ideal_hist[:, :(self.timestep + 1)]
        ideal_dist = mean_item_dist_pairs(pairs, ideal_hist, self.ideal_item_attrs)
        this_dist = mean_item_dist_pairs(pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(this_dist - ideal_dist)
        self.timestep += 1 # increment timestep

"""
Custom RS algorithms
"""
class RandomRecommender(ContentFiltering):
    """
    Random recommender - randomly update user representation at every step
    """
    def _update_internal_state(self, interactions):
        self.items_hat.value[:, :] = self.random_state.random(self.items_hat.shape)
        self.users_hat.value[:, :] = self.random_state.random(self.users_hat.shape)

    def process_new_items(self, new_items):
        """
        Generate random attributes for new items.
        """
        num_items = new_items.shape[1]
        num_attr = self.items_hat.value.shape[0]
        item_representation = self.random_state.random((num_attr, num_items))
        return item_representation

class IdealRecommender(ContentFiltering):
    """
    With the Ideal Recommender, we make the *strong assumption* that the true scores are provided
    to the recommender system through a custom scoring function, which always returns the true
    underlying user-item scores. Therefore, this class is pretty much an empty skeleton; the only
    modification is that we don't update any internal state of the recommender at each time step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_internal_state(self, interactions):
        # do not change users_hat!
        pass

class ChaneyContent(ContentFiltering):
    """
    Chaney ContentFiltering model which uses NNLS solver
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_internal_state(self, interactions):
        # update cumulative interactions
        num_new_items = self.items_hat.shape[1] - self.cumulative_interactions.shape[1] # how many new items were added to the system?
        if num_new_items > 0:
            self.cumulative_interactions = np.hstack([self.cumulative_interactions, np.zeros((self.num_users, num_new_items))]) # add new items to cumulative interactions
        self.cumulative_interactions[self.users.user_vector, interactions] += 1

    def train(self):
        if hasattr(self, 'cumulative_interactions') and self.cumulative_interactions.sum() > 0: # if there are interactions present:
            items_to_train = self.cumulative_interactions.shape[1] # can't train representations for new items before interactions have happened!
            for i in range(self.num_users):
                item_attr = self.items_hat.value[:, :items_to_train].T
                self.users_hat.value[i, :] = nnls(item_attr, self.cumulative_interactions[i, :])[0] # solve for Content Filtering representation using nnls solver
            num_new_items = self.items_hat.shape[1] - self.cumulative_interactions.shape[1] # how many new items were added to the system?

        else:
            self.cumulative_interactions = np.zeros((self.users_hat.shape[0], self.items_hat.shape[1]))
        super().train()
