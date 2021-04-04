import trecs
import numpy as np
import matplotlib.pyplot as plt
from trecs.models import ContentFiltering, PopularityRecommender, ImplicitMF, SocialFiltering
from trecs.components import Users, Items, Creators, ActualUserScores
from trecs.metrics import InteractionSimilarity, Measurement
from trecs.random import Generator
from collections import defaultdict
from chaney_utils import (
    gen_social_network, 
    mu_sigma_to_alpha_beta, 
    exclude_new_items, 
    perfect_scores,
    interleave_new_items,
    process_measurement,
    MeanDistanceSimUsers,
    MeanInteractionDistance,
    SimilarUserInteractionSimilarity,
    InteractionTracker,
    RandomRecommender,
    ChaneyContent
)
import argparse
import os
import errno
import warnings
import pprint
import pickle as pkl
warnings.simplefilter("ignore")

class ChaneyCreators(Creators):
    def __init__(self, items_per_creator, learning_rate=0.05, **kwargs):
        """ 
        TODO: documentation
        """
        # all creators make items at every iteration
        self.items_per_creator = items_per_creator
        self.learning_rate = learning_rate
        super().__init__(**kwargs)
        self.rng = Generator(seed=self.seed)
        self.ordered_creator_ids = np.array([])
        
    def generate_items(self):
        """ 
        All content creators create at every step
        """
        num_creators, num_attrs = self.actual_creator_profiles.shape
        creator_mask = self.rng.binomial(
            1, self.creation_probability, self.actual_creator_profiles.shape[0]
        )
        chosen_profiles = np.nonzero(creator_mask)[0].astype(int)  # keep track of who created the items
        self.ordered_creator_ids = np.append(
            self.ordered_creator_ids, np.repeat(chosen_profiles, self.items_per_creator)
        )
        items = np.zeros((self.items_per_creator * num_creators, num_attrs))
        # TODO: ADD BACK IN
        for idx, c in enumerate(chosen_profiles):
            # generate item from creator
            next_idx = (idx+1) * self.items_per_creator
            # 0.1 multiplier helps maintain sparsity
            items[idx:next_idx, :] = self.rng.dirichlet(self.actual_creator_profiles[c, :] * 0.1, size=self.items_per_creator)
        return items.T
    
    def update_profiles(self, interactions, items):
        """
        Update each creator's profile by the items that gained interaction
        that were made by that creator.

        Parameters
        -----------

            interactions: numpy.ndarray or list
                A matrix where row `i` corresponds to the attribute vector
                that user `i` interacted with.
        """
        # total number of items should be equal to length of ordered_creator_ids
        assert len(self.ordered_creator_ids) == items.shape[1]
        # collapse interactions
        item_ids = interactions.reshape(-1) # readjust indices
        creators_to_update = self.ordered_creator_ids[item_ids].astype(int)
        # note that we remove the last attribute
        weighted_items = (self.learning_rate * items[:, item_ids]).T

        # update the rows at index creators_to_update by adding
        # the rows corresponding to the items they created
        np.add.at(self.actual_creator_profiles, (creators_to_update, slice(None)), weighted_items)
        # normalize creators
        self.actual_creator_profiles /= self.actual_creator_profiles.sum(axis=1)[:, np.newaxis]
        self.actual_creator_profiles = np.clip(self.actual_creator_profiles, 1e-4, 1) # prevent values from getting too low
        
# generates user scores on the fly
def ideal_content_score_fns(sigma, mu_n, num_items_per_iter, generator):
    """
    This is the scoring function used for the Ideal Recommender when content creators are introduced.
    This is necessary because scores are generated for items on the fly, rather than being generated
    at the beginning of the simulation.
    
    Returns the score function for the model and the score function for the users.
    """
    alpha, beta = mu_sigma_to_alpha_beta(mu_n, sigma)
    # start with empty array
    true_user_item_utils = None
    def model_score_fn(user_attrs, item_attrs):
        # generate utils for new items
        nonlocal true_user_item_utils
        num_users, num_items = user_attrs.shape[0], item_attrs.shape[1]
        if true_user_item_utils is None:
            # initialize empty user item score array
            true_user_item_utils = np.array([]).reshape((num_users, num_items))
        if item_attrs.shape[1] == num_items_per_iter: # EDGE CASE: when num_items_per_iter = num_items
            true_utils_mean = user_attrs @ item_attrs
            true_utils_mean = np.clip(true_utils_mean, 1e-9, None)
            user_alphas, user_betas = mu_sigma_to_alpha_beta(true_utils_mean, sigma)
            true_utility = generator.beta(user_alphas, user_betas, size=(num_users, num_items))
            true_user_item_utils = np.hstack((true_user_item_utils, true_utility))
            # all predicted scores for these "new" items will be negative infinity,
            # ensuring they never get recommended
            return np.ones(true_user_item_utils.shape) * float('-inf')
        else:
            # assume this is when train() was called on the entire item set
            assert (num_users, num_items) == true_user_item_utils.shape
            return true_user_item_utils[:num_users, :num_items].copy() # subset to correct dimensions
        
    def user_score_fn(user_profiles, item_attributes):
        """
        The function that calculates user scores depends on the true utilities,
        which are generated during the `model_score_fn` step.
        """
        nonlocal alpha
        nonlocal beta
        num_items = item_attributes.shape[1]
        num_users = user_profiles.shape[0]
        # calculate percentage of utility known
        perc_util_known = generator.beta(alpha, beta, size=(num_users, num_items))
        # get the true utilities for the items that were just created;
        # they should be the most recently added items to the user-item
        # score matrix - this is only true for the ideal recommender
        true_utility = true_user_item_utils[:, -num_items:]
        known_utility = true_utility * perc_util_known
        return known_utility
    
    return model_score_fn, user_score_fn

class IdealContentRecommender(ContentFiltering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _update_internal_state(self, interactions):
        # do not change users_hat! 
        pass

    
def user_score_fn(rec, mu_n, sigma, generator):
    alpha, beta = mu_sigma_to_alpha_beta(mu_n, sigma)
    def score_fn(user_profiles, item_attributes):
        nonlocal alpha
        nonlocal beta
        num_items = item_attributes.shape[1]
        num_users = user_profiles.shape[0]
        # calculate percentage of utility known
        perc_util_known = generator.beta(alpha, beta, size=(num_users, num_items))
        true_utils_mean = user_profiles @ item_attributes
        true_utils_mean = np.clip(true_utils_mean, 1e-9, None)
        user_alphas, user_betas = mu_sigma_to_alpha_beta(true_utils_mean, sigma)
        true_utility = generator.beta(user_alphas, user_betas, size=(num_users, num_items))
        known_utility = true_utility * perc_util_known
        return known_utility
        
    return score_fn

def sample_users_and_creators(rng, num_users, num_creators, num_attrs, num_sims):
    # multiplier of *10 comes from Chaney paper
    user_params = rng.dirichlet(np.ones(num_attrs), size=num_sims) * 10

    # each element in users is the users vector in one simulation
    users, creators, social_networks = [], [], []
    for sim_index in range(num_sims):
        # generate user preferences and item attributes
        user_prefs = rng.dirichlet(user_params[sim_index, :], size=num_users) # 100 users
        creator_attrs = rng.dirichlet(np.ones(num_attrs) * 10, size=num_creators)

        # add all synthetic data to list
        users.append(user_prefs) 
        social_networks.append(gen_social_network(user_prefs))
        creators.append(creator_attrs) 
        
    return users, creators, social_networks

def init_sim_state(user_profiles, creator_profiles, rng, arg_dict):
    # each user interacts with items based on their (noisy) knowledge of their own scores
    # user choices also depend on the order of items they are recommended
    u = Users(
        actual_user_profiles=user_profiles, 
        size=(arg_dict["num_users"], arg_dict["num_attrs"]), 
        num_users=arg_dict["num_users"], 
        attention_exp=arg_dict["attention_exp"], 
        repeat_interactions=False
    )
    c = ChaneyCreators(
        arg_dict["items_per_creator"],
        actual_creator_profiles=creator_profiles.copy(),
        creation_probability=1.0, 
        learning_rate=arg_dict["learning_rate"]
    )
    empty_item_set = np.array([]).reshape((arg_dict["num_attrs"], 0)) # initialize empty item set
    
    init_params = {
        "num_items_per_iter": arg_dict["new_items_per_iter"],
        "num_users": arg_dict["num_users"],
        "num_items": 0, # all simulations start with 0 items
        "interleaving_fn": interleave_new_items(rng),
    }

    run_params = {
        "random_items_per_iter": arg_dict["new_items_per_iter"],
        "vary_random_items_per_iter": False, # exactly X items are interleaved
    }

    return u, c, empty_item_set, init_params, run_params

"""
Methods for running each type of simulation
"""
def run_ideal_sim(user_prefs, creator_profiles, pairs, arg_dict, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, creator_profiles, rng, arg_dict)
    if arg_dict["repeated_training"]:
        post_startup_rec_size = "all"
    else: # only serve items that were in the initial training set
        post_startup_rec_size = arg_dict["total_items_in_startup"] + arg_dict["new_items_per_iter"] # show all items from training set plus interleaved itmes
    model_score_fn, user_score_fn = ideal_content_score_fns(arg_dict["sigma"], arg_dict["mu_n"], arg_dict["new_items_per_iter"], rng)
    ideal = IdealContentRecommender(
        user_representation=user_prefs,
        creators=c, 
        actual_user_representation=u,
        actual_item_representation=empty_items,
        score_fn=model_score_fn,
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    ideal.users.set_score_function(user_score_fn)
    ideal.add_metrics(MeanInteractionDistance(pairs), InteractionTracker())
    ideal.startup_and_train(timesteps=arg_dict["startup_iters"])
    ideal.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    ideal.run(timesteps=arg_dict["sim_iters"], train_between_steps=arg_dict["repeated_training"], **run_params)
    ideal.close()
    return ideal

def run_content_sim(user_prefs, creator_profiles, pairs, ideal_interactions, ideal_item_attrs, arg_dict, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, creator_profiles, rng, arg_dict)
    if not arg_dict["repeated_training"]:
        post_startup_rec_size = arg_dict["total_items_in_startup"] + arg_dict["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    chaney = ChaneyContent(
        creators=c, 
        num_attributes=arg_dict["num_attrs"], 
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(arg_dict["new_items_per_iter"]),
        **init_params)
    # set score function here because it requires a reference to the recsys
    chaney.users.set_score_function(user_score_fn(chaney, arg_dict["mu_n"], arg_dict["sigma"], rng))
    metrics = [
        MeanInteractionDistance(pairs), 
        MeanDistanceSimUsers(ideal_interactions, ideal_item_attrs)
    ]
    chaney.add_metrics(*metrics)
    chaney.add_state_variable(chaney.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    chaney.startup_and_train(timesteps=arg_dict["startup_iters"]) # update user representations, but only serve random items
    chaney.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    chaney.run(timesteps=arg_dict["sim_iters"], train_between_steps=arg_dict["repeated_training"], **run_params)
    chaney.close() # end logging
    return chaney
    
def run_mf_sim(user_prefs, creator_profiles, pairs, ideal_interactions, ideal_item_attrs, arg_dict, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, creator_profiles, rng, arg_dict)
    if not arg_dict["repeated_training"]:
        post_startup_rec_size = arg_dict["total_items_in_startup"] + arg_dict["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    mf = ImplicitMF(
        creators=c,
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        num_latent_factors=arg_dict["num_attrs"],
        score_fn=exclude_new_items(arg_dict["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    mf.users.set_score_function(user_score_fn(mf, arg_dict["mu_n"], arg_dict["sigma"], rng))
    metrics = [
        MeanInteractionDistance(pairs), 
        MeanDistanceSimUsers(ideal_interactions, ideal_item_attrs)
    ]
    mf.add_metrics(*metrics)
    mf.add_state_variable(mf.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    mf.startup_and_train(timesteps=arg_dict["startup_iters"], no_new_items=False) # update user representations, but only serve random items
    mf.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    mf.run(timesteps=arg_dict["sim_iters"], train_between_steps=arg_dict["repeated_training"], reset_interactions=False, **run_params)
    mf.close() # end logging
    return mf
    
    
def run_sf_sim(social_network, user_prefs, creator_profiles, pairs, ideal_interactions, ideal_item_attrs, arg_dict, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, creator_profiles, rng, arg_dict)
    if not arg_dict["repeated_training"]:
        post_startup_rec_size = arg_dict["total_items_in_startup"] + arg_dict["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    sf = SocialFiltering(
        creators=c,
        user_representation=social_network, 
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(arg_dict["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    sf.users.set_score_function(user_score_fn(sf, arg_dict["mu_n"], arg_dict["sigma"], rng))
    metrics = [
        MeanInteractionDistance(pairs), 
        MeanDistanceSimUsers(ideal_interactions, ideal_item_attrs)
    ]
    sf.add_metrics(*metrics)
    sf.add_state_variable(sf.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    sf.startup_and_train(timesteps=arg_dict["startup_iters"])
    sf.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    sf.run(timesteps=arg_dict["sim_iters"], train_between_steps=arg_dict["repeated_training"], **run_params)
    sf.close() # end logging
    return sf

def run_pop_sim(user_prefs, creator_profiles, pairs, ideal_interactions, ideal_item_attrs, arg_dict, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, creator_profiles, rng, arg_dict)
    if not arg_dict["repeated_training"]:
        post_startup_rec_size = arg_dict["total_items_in_startup"] + arg_dict["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    p = PopularityRecommender(
        creators=c,
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(arg_dict["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    p.users.set_score_function(user_score_fn(p, arg_dict["mu_n"], arg_dict["sigma"], rng))
    metrics = [
        MeanInteractionDistance(pairs), 
        MeanDistanceSimUsers(ideal_interactions, ideal_item_attrs)
    ]
    p.add_metrics(*metrics)
    p.add_state_variable(p.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    p.startup_and_train(timesteps=arg_dict["startup_iters"])
    p.set_num_items_per_iter(post_startup_rec_size)
    p.run(timesteps=arg_dict["sim_iters"], train_between_steps=arg_dict["repeated_training"], **run_params)
    p.close() # end logging
    return p
    
def run_random_sim(user_prefs, creator_profiles, pairs, ideal_interactions, ideal_item_attrs, arg_dict, rng):
    u, c, empty_items, init_params, run_params = init_sim_state(user_prefs, creator_profiles, rng, arg_dict)
    if not arg_dict["repeated_training"]:
        post_startup_rec_size = arg_dict["total_items_in_startup"] + arg_dict["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    r = RandomRecommender(
        creators=c,
        num_attributes=arg_dict["num_attrs"],
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(arg_dict["new_items_per_iter"]),
        **init_params
    )
    # set score function here because it requires a reference to the recsys
    r.users.set_score_function(user_score_fn(r, arg_dict["mu_n"], arg_dict["sigma"], rng))
    metrics = [
        MeanInteractionDistance(pairs), 
        MeanDistanceSimUsers(ideal_interactions, ideal_item_attrs)
    ]
    r.add_metrics(*metrics) # random pairing of users
    r.add_state_variable(r.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    r.startup_and_train(timesteps=arg_dict["startup_iters"])
    r.set_num_items_per_iter(post_startup_rec_size)
    # always train between steps so items are randomly scored
    r.run(timesteps=arg_dict["sim_iters"], train_between_steps=True, **run_params)
    r.close() # end logging
    return r

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running content creator simulations')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--num_creators', type=int, default=10)
    parser.add_argument('--num_attrs', type=int, default=20)
    parser.add_argument('--num_sims', type=int, default=25)
    parser.add_argument('--startup_iters', type=int, required=True)
    parser.add_argument('--sim_iters', type=int, required=True)
    parser.add_argument('--new_items_per_iter', type=int, default=10)
    parser.add_argument('--repeated_training', type=bool, required=True)
    parser.add_argument('--items_per_creator', type=int, default=1)
    parser.add_argument('--attention_exp', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--mu_n', type=float, default=0.98)
    parser.add_argument('--sigma', type=float, default=1e-5)

    args = parser.parse_args()
    arg_dict = vars(args)
    
    # define ancillary variables
    arg_dict["new_items"] = arg_dict["items_per_creator"] * arg_dict["num_creators"]
    arg_dict["total_items_in_startup"] = arg_dict["new_items"] * arg_dict["startup_iters"]
    
    print("Creating experiment output folder... 💻")
    # create output folder
    try:
        os.makedirs(arg_dict["output_dir"])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    # write experiment arguments to file
    with open(os.path.join(arg_dict["output_dir"], "args.txt"), "w") as log_file:
        pprint.pprint(arg_dict, log_file)
    
    rng = np.random.default_rng(arg_dict["seed"])
    
    # sample initial user / creator profiles
    print("Sampling initial user and creator profiles... 🔬")
    users, creators, social_networks = sample_users_and_creators(
        rng, 
        arg_dict["num_users"], 
        arg_dict["num_creators"], 
        arg_dict["num_attrs"], 
        arg_dict["num_sims"]
    )
    
    # run simulations
    model_keys = ["ideal", "content_chaney", "mf", "sf", "popularity", "random"]
    # stores results for each type of model for each type of user pairing (random or cosine similarity)
    result_metrics = {"mean_item_dist": defaultdict(list), "sim_user_dist": defaultdict(list)}
    models = {} # temporarily stores models

    print("Running simulations...👟")
    for i in range(arg_dict["num_sims"]):
        true_prefs = users[i] # underlying true preferences
        creator_profiles = creators[i]
        social_network = social_networks[i]

        # generate random pairs for evaluating jaccard similarity
        pairs = [rng.choice(arg_dict["num_users"], 2, replace=False) for _ in range(800)]

        models["ideal"] = run_ideal_sim(true_prefs, creator_profiles, pairs, arg_dict, rng)
        ideal_interactions = np.hstack(process_measurement(models["ideal"], "interaction_history")) # pull out the interaction history for the ideal simulations
        ideal_attrs = models["ideal"].actual_item_attributes
        models["content_chaney"] = run_content_sim(true_prefs, creator_profiles, pairs, ideal_interactions, ideal_attrs, arg_dict, rng)
        models["mf"] = run_mf_sim(true_prefs, creator_profiles, pairs, ideal_interactions, ideal_attrs, arg_dict, rng)
        models["sf"] = run_sf_sim(social_network, true_prefs, creator_profiles, pairs, ideal_interactions, ideal_attrs, arg_dict, rng)
        models["popularity"] = run_pop_sim(true_prefs, creator_profiles, pairs, ideal_interactions, ideal_attrs, arg_dict, rng)
        models["random"] = run_random_sim(true_prefs, creator_profiles, pairs, ideal_interactions, ideal_attrs, arg_dict, rng)

        # extract results from each model
        for model_key in model_keys:
            model = models[model_key]
            result_metrics["mean_item_dist"][model_key].append(process_measurement(model, "mean_interaction_dist"))
            if model_key is not "ideal": # homogenization of similar users is always measured relative to the ideal model
                result_metrics["sim_user_dist"][model_key].append(process_measurement(model, "sim_user_dist"))
    
    # write results ot pickle file
    output_file = os.path.join(arg_dict["output_dir"], "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file, "wb"), -1)
    print("Done with simulation! 🎉")
    