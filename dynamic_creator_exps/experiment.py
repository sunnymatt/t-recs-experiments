from trecs.components import Creators, Users, Items
from trecs.models import ContentFiltering, PopularityRecommender
import numpy as np
from collections import defaultdict
import pickle as pkl
import datetime
import configargparse

# import utils from parent directory
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils import create_profiles, calc_group_preferences, cos_sim_score_fn
from dynamic_creators import DynamicCreators
from custom_metrics import CreatorAvgGroupSkew
from custom_rec import RandomRecommender

WORDS = pkl.load(open("words.pkl", "rb"))


def generate_folder_name():
    """Generate a human-readable folder name to store the results of the experiment."""
    return "-".join(np.random.choice(WORDS, 3, replace=False))


# create folder for where experimental results will be stored and store args
folder_name = generate_folder_name()
while os.path.exists(f"./{folder_name}"):
    folder_name = generate_folder_name()

# parse input arguments (optionally, use config)
parser = configargparse.ArgumentParser(description="Run T-RECS simulation.")
parser.add(
    "-c", "--config", required=False, is_config_file=True, help="config file path"
)
parser.add_argument(
    "--steps", type=int, default=10, help="number of timesteps for simulation to run"
)
parser.add_argument(
    "--num_users",
    type=int,
    default=100,
    help="number of timesteps for simulation to run",
)
parser.add_argument(
    "--majority_fraction",
    type=float,
    default=0.8,
    help="number of timesteps for simulation to run",
)
parser.add_argument(
    "--num_seed_items",
    type=int,
    default=10,
    help="number of items to seed simulation with",
)
parser.add_argument(
    "--num_seed_majority_items",
    type=int,
    default=5,
    help="number of seed items that come from majority group",
)
parser.add_argument(
    "--num_creators",
    type=int,
    default=100,
    help="number of content creators in the system",
)
parser.add_argument(
    "--creation_prob",
    type=float,
    default=0.5,
    help="probability that an individual creator will create an item at a timestep",
)
parser.add_argument(
    "--num_attrs",
    type=int,
    default=10,
    help="number of attributes per item / attributes per user profile",
)
parser.add_argument(
    "--collapse_group_attrs",
    action="store_true",
    help="collapse group attributes into a single ",
)
parser.add_argument(
    "--group_strength",
    type=float,
    default=4.0,
    help="weight factor for how much group membership influences user preference",
)
parser.add_argument(
    "--creator_learning_rate", type=float, default=0.01, help="creator learning rate"
)
parser.add_argument(
    "--item_bias",
    type=float,
    default=-0.5,
    help="additive bias applied to all item attributes; used to balance",
)
parser.add_argument(
    "--startup_iters",
    type=int,
    default=20,
    help="number of startup iterations of random recommendations",
)
parser.add_argument(
    "--items_per_iter",
    type=int,
    default=10,
    help="number of items recommended to each user at each iteration",
)
parser.add_argument(
    "--trials", type=int, default=20, help="number of trials to repeat simulation"
)
parser.add_argument(
    "--model",
    type=str,
    choices=["cf", "pop", "random"],
    default="cf",
    help="recommender model type",
)

args = parser.parse_args()

# write args to new config
args_dict = vars(args)
os.mkdir(f"./{folder_name}")
with open(f"./{folder_name}/source_args.txt", "w") as out:
    print(parser.format_values(), file=out)
parser.write_config_file(args, [f"./{folder_name}/args.yml"])

# redirect stdout and stderr to a log file
sys.stdout = open(f"./{folder_name}/out.log", 'w')
sys.stderr = sys.stdout

# parameter initialization
# most of the parameter initialization is handled by arg_parse
NUM_MAJORITY = round(args.majority_fraction * args.num_users)
SEPARATE_ATTR = not args.collapse_group_attrs
model_dict = {
    "cf": ContentFiltering,
    "pop": PopularityRecommender,
    "random": RandomRecommender,
}

creator_profs = defaultdict(list)

for i in range(args.trials):
    print(f"Current trial: {i+1} out of {args.trials}")
    user_profiles, item_profiles, creator_profiles = create_profiles(
        total_users=args.num_users,
        num_majority_users=NUM_MAJORITY,
        total_items=args.num_seed_items,
        num_majority_items=args.num_seed_majority_items,
        total_creators=args.num_creators,
        dynamic_creators=True,
        creation_probability=args.creation_prob,
        num_attrs=args.num_attrs,
        separate_attr=SEPARATE_ATTR,
        group_strength=args.group_strength,
        creator_learning_rate=args.creator_learning_rate,
        item_bias=args.item_bias,  # IMPORTANT: this ensures generated items will be between -0.5 and 0.5
    )
    rec = model_dict[args.model](
        actual_item_representation=item_profiles,
        item_representation=item_profiles,
        actual_user_representation=Users(actual_user_profiles=user_profiles),
        creators=creator_profiles,
        num_items_per_iter=args.items_per_iter,
        verbose=False,
    )
    if args.startup_iters > 0:
        rec.startup_and_train(timesteps=args.startup_iters)

    # record creator profiles after CF algorithm

    rec.run(
        timesteps=args.steps, random_items_per_iter=4, vary_random_items_per_iter=True
    )
    mean_creator = creator_profiles.actual_creator_profiles.mean(axis=0).reshape(1, -1)
    creator_profs[args.steps].append(mean_creator)


pkl.dump(creator_profs, open(f"./{folder_name}/creator_profs.pkl", "wb"))
