import numpy as np
from trecs.components import Creators
from trecs.random import Generator

class DynamicCreators(Creators):
    """
    Dynamic creators update their attributes in response to the user
    interactions with items.
    """
    def __init__(
        self,
        init_items = 0,
        actual_creator_profiles=None,
        creation_probability=0.5,
        size=None,
        verbose=False,
        seed=None,
        learning_rate=0.05,
    ):
        Creators.__init__(
            self,
            actual_creator_profiles,
            creation_probability,
            size,
            verbose,
            seed,
        )
        self.learning_rate = learning_rate
        # keep track of the order in which creators made items
        self.ordered_creator_ids = np.array([])
        self.init_items = init_items # number of items the system starts with

    def generate_items(self):
        """
        Generates new items. Each creator probabilistically creates a new item.
        Item attributes are generated using each creator's profile
        as a series of Bernoulli random variables. Therefore, item attributes
        will be binarized. To change this behavior, simply write a custom
        class that overwrites this method.

        Returns
        ---------
            A numpy matrix of dimension :math:`|I_n|\\times|A|`, where
            :math:`|I_n|` represents the number of new items, and :math:`|A|`
            represents the number of attributes for each item.
        """
        # Generate mask by tossing coin for each creator to determine who is releasing content
        # This should result in a _binary_ matrix of size (num_creators,)
        if (self.actual_creator_profiles < 0).any() or (self.actual_creator_profiles > 1).any():
            raise ValueError("Creator profile attributes must be between zero and one.")
        creator_mask = Generator(seed=self.seed).binomial(
            1, self.creation_probability, self.actual_creator_profiles.shape[0]
        )

        chosen_profiles = self.actual_creator_profiles[creator_mask == 1, :]

        # keep track of who created at which step
        self.ordered_creator_ids = np.append(
            self.ordered_creator_ids, np.nonzero(creator_mask)
        )


        # for each creator that will add new items, generate Bernoulli trial
        # for item attributes
        items = Generator(seed=self.seed).binomial(
            1, chosen_profiles.reshape(-1), chosen_profiles.size
        ).reshape(chosen_profiles.shape)

        # add attribute for the opposite group
        last_attr = (1 - items[:, -1]).reshape(-1, 1)
        return np.hstack((items, last_attr)).T

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
        # change group attributes to negative 1 and 1 so it's easier
        # to update creator attributes
        items[items == 0] = -1
        # collapse interactions
        item_ids = interactions.reshape(-1) - self.init_items # readjust indices
        item_ids = item_ids[item_ids >= 0]
        creators_to_update = self.ordered_creator_ids[item_ids].astype(int)
        interacted_items = (item_ids + self.init_items).astype(int)
        # note that we remove the last attribute
        weighted_items = (self.learning_rate * items[:-1, interacted_items]).T

        # update the rows at index creators_to_update by adding
        # the rows corresponding to the items they created
        np.add.at(self.actual_creator_profiles, (creators_to_update, slice(None)), weighted_items)

        # creator profiles should be between zero and one
        self.actual_creator_profiles[self.actual_creator_profiles < 0 ] = 0
        self.actual_creator_profiles[self.actual_creator_profiles > 1 ] = 1

        # reset items to be back to normal
        items[items == -1] = 0

