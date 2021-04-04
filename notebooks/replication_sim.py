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
    ChaneyContent,
    IdealRecommender
)
import argparse
import os
import errno
import warnings
import pprint
import pickle as pkl
warnings.simplefilter("ignore")


class NewItemFactory(Creators):
    """
    This custom "content creator" pumps out new items at every iteration. It starts with the full 
    knowledge of all items at the beginning of the simulation, and then as the simulation progresses,
    "releases" new items to the RS. Requires all items to be passed in during initialization.
    """
    def __init__(self, items, items_per_iteration):
        """ Expects items in matrix of dimension |A| x |I|, where
            |A| is the number of attributes and |I| is the number
            of items.
        """
        self.items = items
        self.items_per_iteration = items_per_iteration
        self.idx = 0
        
    def generate_items(self):
        """ The items generated are simply selected from a particular
            range of indices, which gets incremented at each timestep.
        """
        if self.idx > self.items.shape[1]:
            raise RuntimeError("Ran out of items to generate!")
        idx_start = self.idx
        idx_end = self.idx + self.items_per_iteration
        self.idx = idx_end
        return self.items[:, idx_start:idx_end]
    
class ChaneyUsers(Users):
    """
    This special subclass of `Users` allows users to know their true scores for all items
    (even those that haven't been created yet). Requires these true scores to be
    passed in at initialization.
    """
    def __init__(self, true_scores, *args, **kwargs):
        self.true_scores = np.copy(true_scores) # contains all user-item scores
        super().__init__(*args, **kwargs)
        
    def compute_user_scores(self, items):
        """ No need to do this at initialization - user will be set with scores later.
        """
        self.actual_user_scores = ActualUserScores(self.true_scores)
        
    def score_new_items(self, items):
        """ Chaney users are special - they already know all their
            utility values for all items
        """
        pass
    
    def get_user_feedback(self, *args, **kwargs):
        interactions = super().get_user_feedback(*args, **kwargs)
        return interactions
    
    
"""
Functions for initializing and running simulations
"""

def init_sim_state(known_scores, all_items, arg_dict, rng):
    # each user interacts with items based on their (noisy) knowledge of their own scores
    # user choices also depend on the order of items they are recommended
    u = ChaneyUsers(
        np.copy(known_scores), 
        size=(arg_dict["num_users"], arg_dict["num_attrs"]), 
        num_users=arg_dict["num_users"], 
        attention_exp=arg_dict["attention_exp"],
        repeat_interactions=False
    )
    item_factory = NewItemFactory(np.copy(all_items), arg_dict["new_items_per_iter"])
    empty_item_set = np.array([]).reshape((arg_dict["num_attrs"], 0)) # initialize empty item set
    
    # simpler way to pass common arguments to simulations
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
    
    return u, item_factory, empty_item_set, init_params, run_params


def run_ideal_sim(user_prefs, item_attrs, true_utils, noisy_utilities, pairs, args, rng):
    u, item_factory, empty_items, init_params, run_params = init_sim_state(noisy_utilities, item_attrs, args, rng)
    if args["repeated_training"]:
        post_startup_rec_size = "all"
    else: # only serve items that were in the initial training set
        post_startup_rec_size = (args["startup_iters"] + 1) + args["new_items_per_iter"] # show all items from training set plus interleaved items
    ideal = IdealRecommender(
        user_representation=user_prefs,
        creators=item_factory, 
        actual_user_representation=u,
        actual_item_representation=empty_items,
        score_fn=perfect_scores(args["new_items_per_iter"], true_utils),
        **init_params
    )
    ideal.add_metrics(InteractionSimilarity(pairs), MeanInteractionDistance(pairs), InteractionTracker())
    ideal.startup_and_train(timesteps=args["startup_iters"])
    ideal.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    ideal.close() # end logging
    ideal.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    return ideal

def run_content_sim(item_attrs, noisy_utilities, pairs, ideal_interactions, args, rng):
    u, item_factory, empty_items, init_params, run_params = init_sim_state(noisy_utilities, item_attrs, args, rng)
    if not args["repeated_training"]:
        post_startup_rec_size = (args["startup_iters"] + 1) + args["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    chaney = ChaneyContent(
        creators=item_factory, 
        num_attributes=args["num_attrs"], 
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params)
    metrics = [
        InteractionSimilarity(pairs), 
        MeanInteractionDistance(pairs), 
        SimilarUserInteractionSimilarity(ideal_interactions),
        MeanDistanceSimUsers(ideal_interactions, item_attrs)
    ]
    chaney.add_metrics(*metrics)
    chaney.add_state_variable(chaney.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    chaney.startup_and_train(timesteps=args["startup_iters"]) # update user representations, but only serve random items
    chaney.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    chaney.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    chaney.close() # end logging
    return chaney
    
def run_mf_sim(item_attrs, noisy_utilities, pairs, ideal_interactions, args, rng):
    u, item_factory, empty_items, init_params, run_params = init_sim_state(noisy_utilities, item_attrs, args, rng)
    if not args["repeated_training"]:
        post_startup_rec_size = (args["startup_iters"] + 1) + args["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    mf = ImplicitMF(
        creators=item_factory,
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        num_latent_factors=args["num_attrs"],
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    metrics = [
        InteractionSimilarity(pairs), 
        MeanInteractionDistance(pairs), 
        SimilarUserInteractionSimilarity(ideal_interactions),
        MeanDistanceSimUsers(ideal_interactions, item_attrs)
    ]
    mf.add_metrics(*metrics)
    mf.add_state_variable(mf.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    mf.startup_and_train(timesteps=args["startup_iters"], no_new_items=False) # update user representations, but only serve random items
    mf.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    mf.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], reset_interactions=False, **run_params)
    mf.close() # end logging
    return mf
    
    
def run_sf_sim(social_network, item_attrs, noisy_utilities, pairs, ideal_interactions, args, rng):
    u, item_factory, empty_items, init_params, run_params = init_sim_state(noisy_utilities, item_attrs, args, rng)
    if not args["repeated_training"]:
        post_startup_rec_size = (args["startup_iters"] + 1) + args["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    sf = SocialFiltering(
        creators=item_factory,
        user_representation=social_network, 
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    metrics = [
        InteractionSimilarity(pairs), 
        MeanInteractionDistance(pairs), 
        SimilarUserInteractionSimilarity(ideal_interactions),
        MeanDistanceSimUsers(ideal_interactions, item_attrs)
    ]
    sf.add_metrics(*metrics)
    sf.add_state_variable(sf.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    sf.startup_and_train(timesteps=args["startup_iters"])
    sf.set_num_items_per_iter(post_startup_rec_size) # show all items from training set plus interleaved itmes
    sf.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    sf.close() # end logging
    return sf

def run_pop_sim(item_attrs, noisy_utilities, pairs, ideal_interactions, args, rng):
    u, item_factory, empty_items, init_params, run_params = init_sim_state(noisy_utilities, item_attrs, args, rng)
    if not args["repeated_training"]:
        post_startup_rec_size = (args["startup_iters"] + 1) + args["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    p = PopularityRecommender(
        creators=item_factory,
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    metrics = [
        InteractionSimilarity(pairs), 
        MeanInteractionDistance(pairs), 
        SimilarUserInteractionSimilarity(ideal_interactions),
        MeanDistanceSimUsers(ideal_interactions, item_attrs)
    ]
    p.add_metrics(*metrics)
    p.add_state_variable(p.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    p.startup_and_train(timesteps=args["startup_iters"])
    p.set_num_items_per_iter(post_startup_rec_size)
    p.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"], **run_params)
    p.close() # end logging
    return p
    
def run_random_sim(item_attrs, noisy_utilities, pairs, ideal_interactions, args, rng):
    u, item_factory, empty_items, init_params, run_params = init_sim_state(noisy_utilities, item_attrs, args, rng)
    if not args["repeated_training"]:
        post_startup_rec_size = (args["startup_iters"] + 1) + args["new_items_per_iter"] # show all items from training set plus interleaved items
    else: # only serve items that were in the initial trainin gset
        post_startup_rec_size = "all"
    r = RandomRecommender(
        creators=item_factory,
        num_attributes=args["num_attrs"],
        actual_item_representation=empty_items, 
        actual_user_representation=u,
        score_fn=exclude_new_items(args["new_items_per_iter"]),
        **init_params
    )
    metrics = [
        InteractionSimilarity(pairs), 
        MeanInteractionDistance(pairs), 
        SimilarUserInteractionSimilarity(ideal_interactions),
        MeanDistanceSimUsers(ideal_interactions, item_attrs)
    ]
    r.add_metrics(*metrics) # random pairing of users
    r.add_state_variable(r.users_hat) # need to do this so that the similarity metric uses the user representation from before interactions were trained
    r.startup_and_train(timesteps=args["startup_iters"])
    r.set_num_items_per_iter(post_startup_rec_size)
    # train between steps because we make the user representations random every time
    r.run(timesteps=args["sim_iters"], train_between_steps=True, **run_params)
    r.close() # end logging
    return r