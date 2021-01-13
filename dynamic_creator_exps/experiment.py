from trecs.components import Creators, Users, Items
from trecs.models import ContentFiltering, PopularityRecommender
import numpy as np
from collections import defaultdict
import pickle as pkl
import pprint
import datetime

# parent directory
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils import create_profiles, calc_group_preferences, cos_sim_score_fn
from dynamic_creators import DynamicCreators
from custom_metrics import CreatorAvgGroupSkew
import argparse

parser = argparse.ArgumentParser(description='Run ContentFiltering simulation.')
parser.add_argument('--steps', type=int, default=10,
                    help='number of timesteps for simulation to run')
parser.add_argument('--num_users', type=int, default=100,
                    help='number of timesteps for simulation to run')
parser.add_argument('--majority_fraction', type=float, default=0.8,
                    help='number of timesteps for simulation to run')
parser.add_argument('--num_seed_items', type=int, default=10,
                    help='number of items to seed simulation with')
parser.add_argument('--num_seed_majority_items', type=int, default=5,
                    help='number of seed items that come from majority group')
parser.add_argument('--num_creators', type=int, default=100,
                    help='number of content creators in the system')
parser.add_argument('--creation_prob', type=float, default=0.5,
                    help='probability that an individual creator will create an item at a timestep')
parser.add_argument('--num_attrs', type=int, default=10,
                    help='number of attributes per item / attributes per user profile')
parser.add_argument('--collapse_group_attrs', action="store_true",
                    help='collapse group attributes into a single ')
parser.add_argument('--group_strength', type=float, default=4.0,
                    help='weight factor for how much group membership influences user preference')
parser.add_argument('--creator_learning_rate', type=float, default=0.01,
                    help='creator learning rate')
parser.add_argument('--item_bias', type=float, default=-0.5,
                    help='additive bias applied to all item attributes; used to balance')
parser.add_argument('--startup_iters', type=int, default=20,
                    help='number of startup iterations of random recommendations')
parser.add_argument('--items_per_iter', type=int, default=10,
                    help='number of items recommended to each user at each iteration')
parser.add_argument('--trials', type=int, default=20,
                    help='number of trials to repeat simulation')

args = parser.parse_args()
# parameter initializaiton
# redundant now that we use argparse, so we should get rid of this and put
# some of it into the function calls 
NUM_USERS = args.num_users
NUM_MAJORITY = round(args.majority_fraction * NUM_USERS)
NUM_SEED_ITEMS = args.num_seed_items
NUM_SEED_ITEM_MAJORITY = args.num_seed_majority_items
NUM_CREATORS = args.num_creators
CREATION_PROB = args.creation_prob
NUM_ATTRS = args.num_attrs
SEPARATE_ATTR = not args.collapse_group_attrs
GROUP_STRENGTH = args.group_strength
CREATOR_LEARNING_RATE = args.creator_learning_rate
ITEM_BIAS = args.item_bias
NUM_STEPS = args.steps
STARTUP_ITER = args.startup_iters
NUM_ITEMS_PER_ITER = args.items_per_iter
NUM_TRIALS = args.trials

# create folder for where experimental results will be stored and store args
args_dict = vars(args)
folder = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
os.mkdir(f'./{folder}')
with open(f'{folder}/args.txt', 'w') as out:
	pprint.pprint(args_dict, stream=out)


creator_profs = defaultdict(list)

for i in range(NUM_TRIALS):
    print(f"Current trial: {i+1} out of {NUM_TRIALS}")
    user_profiles, item_profiles, creator_profiles = create_profiles(
        total_users = NUM_USERS,
        num_majority_users = NUM_MAJORITY,
        total_items = NUM_SEED_ITEMS,
        num_majority_items = NUM_SEED_ITEM_MAJORITY,
        total_creators = NUM_CREATORS,
        dynamic_creators = True,
        creation_probability = CREATION_PROB,
        num_attrs = NUM_ATTRS,
        separate_attr = SEPARATE_ATTR,
        group_strength = GROUP_STRENGTH,
        creator_learning_rate = CREATOR_LEARNING_RATE,
        item_bias= ITEM_BIAS # IMPORTANT: this ensures generated items will be between -0.5 and 0.5
    )
    cf = ContentFiltering(
        actual_item_representation=item_profiles,
        item_representation=item_profiles,
        actual_user_representation=Users(actual_user_profiles=user_profiles),
        creators = creator_profiles,
        num_items_per_iter=NUM_ITEMS_PER_ITER,
        verbose=False
    )
    cf.startup_and_train(timesteps=STARTUP_ITER)
    
    # record creator profiles after CF algorithm
    
    cf.run(timesteps=NUM_STEPS, random_items_per_iter=4, vary_random_items_per_iter=True)
    mean_creator = creator_profiles.actual_creator_profiles.mean(axis=0).reshape(1, -1)
    creator_profs[NUM_STEPS].append(mean_creator)


pkl.dump(creator_profs, open(f'{folder}/creator_profs.pkl', "wb"))
