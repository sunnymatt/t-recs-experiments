import numpy as np
from dynamic_creators import DynamicCreators
from trecs.components import Creators
from trecs.matrix_ops import inner_product

def create_profiles(
    total_users = None,
    total_items = None,
    total_creators = None,
    dynamic_creators = True,
    num_majority_users = None,
    num_majority_items = None,
    group_strength=0.5,
    num_attrs=10,
    separate_attr=True,
    creation_probability=0.5,
    verbose=True,
    creator_learning_rate=0.01,
    binarize_items=False,
    item_bias=0):
    """
    Creates user and items arrays, where users fall into one of two groups (a majority and a minority group)
    and items have an attribute that indicate whether that item was created by a member of the majority
    group or not.

    The `group_strength` attribute corresponds to how much preference users give to items
    created by members of their own group.

    User profiles and item profiles are generated with the same number of attributes as specified by
    `num_attrs`. Each attribute is generated from a uniform distribution between 0 and 1, except
    for the group preference attribute. For users, the group preference attribute is equal to the
    `group_strength` variable, while for items, the group attribute is equal to 1.0 if it was created
    by a member of the majority group and 0.0 if it was created by a member of the minority group.

    Can also be used to generate
    """
    # Create user profiles
    user_profiles = None
    if total_users is not None:
        assert num_majority_users >= 0.5 * total_users
        user_profiles = np.random.uniform(low=-1.0, high=1.0, size=(total_users, num_attrs))
        # Set the group preference attribute
        if separate_attr: # each group gets its own attribute
            user_profiles[:num_majority_users, num_attrs-2] = group_strength # members of group A
            user_profiles[num_majority_users:, num_attrs-2] = 0 # members of group B
            user_profiles[:num_majority_users, num_attrs-1] = 0 # members of group A
            user_profiles[num_majority_users:, num_attrs-1] = group_strength # members of group B
        else:
            user_profiles[:num_majority_users, num_attrs-1] = group_strength # members of group A
            user_profiles[num_majority_users:, num_attrs-1] = -1 * group_strength # members of group B

    # Create item profiles
    item_profiles = None
    if total_items is not None:
        # uniform items
        if not binarize_items:
            item_profiles = np.random.uniform(size=(num_attrs, total_items))
        else:
            item_profiles = np.random.binomial(1, 0.5, size=(num_attrs, total_items))
        # Set group membership attribute for items
        if separate_attr:
            item_profiles[num_attrs-2, :num_majority_items] = 1.0 # created by group A
            item_profiles[num_attrs-2, num_majority_items:] = 0.0 # created by group B
            item_profiles[num_attrs-1, :num_majority_items] = 0.0 # created by group A
            item_profiles[num_attrs-1, num_majority_items:] = 1.0 # created by group B
        else:
            item_profiles[num_attrs-1, :num_majority_items] = 1.0 # created by group A
            item_profiles[num_attrs-1, num_majority_items:] = 0.0 # created by group B
        item_profiles = item_profiles + item_bias

    creator_profiles = None
    if total_creators is not None:
        if dynamic_creators:
            # for now, creators only have 1 group sensitive attribute. it represents
            # the likelihood of the item being in Group A. see dynamic_creators.py for
            # more details
            creator_profiles = DynamicCreators(
                init_items=total_items,
                size=(total_creators, num_attrs - 1),
                creation_probability=creation_probability,
                learning_rate=creator_learning_rate,
                item_bias=item_bias
            )
        else:
            creator_profiles = Creators(
                size=(total_creators, num_attrs),
                creation_probability=creation_probability,
                learning_rate=creator_learning_rate
            )

    # sanity check that prints the number of items generated by Group A
    if verbose:
        if separate_attr:
            print(f"Percentage of items generated by Group A: {sum(item_profiles[num_attrs-2, :]) / item_profiles.shape[1]}")
        else:
            print(f"Percentage of items generated by Group A: {sum(item_profiles[num_attrs-1, :]) / item_profiles.shape[1]}")
    output = [user_profiles, item_profiles, creator_profiles]
    # only return non-None outputs
    return (profiles for profiles in output if profiles is not None)

def calc_group_preferences(user_profiles, item_profiles, num_majority_users, num_majority_items):
    """
    Sample a majority-group item / minority-group item pair 10,000 times. For each sample, we record
    which item a randomly chosen majority group user preferred and which item a randomly chosen
    minority group member preferred.
    """
    a_preferred_a = 0
    b_preferred_a = 0
    for i in range(10000):
        a_id = np.random.randint(0, num_majority_users) # member of group A
        b_id = np.random.randint(num_majority_users, user_profiles.shape[0]) # member of group B
        a_item_id = np.random.randint(0, num_majority_items) # item index of group A-created item
        b_item_id = np.random.randint(num_majority_items, item_profiles.shape[1]) # item index of group B-created item
        if user_profiles[a_id] @ item_profiles[:, a_item_id] > user_profiles[a_id] @ item_profiles[:, b_item_id]:
            a_preferred_a += 1
        if user_profiles[b_id] @ item_profiles[:, a_item_id] > user_profiles[b_id] @ item_profiles[:, b_item_id]:
            b_preferred_a += 1

    print(f"Members of Group A prefer items from group A {(a_preferred_a/10000 * 100):.2f}% of the time.")
    print(f"Members of Group B prefer items from group A {(b_preferred_a/10000 * 100):.2f}% of the time.")


def cos_sim_score_fn(user_profiles, item_attributes):
    return inner_product(user_profiles, item_attributes, normalize_users=True, normalize_items=True)