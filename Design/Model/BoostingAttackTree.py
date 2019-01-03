"""
BoostingAttackTree model:
    This module contains boostingattacktree based weight boosing estimators
    and many relevant tools.


The module structure is the following:
    -- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
       for all the estimators in the module.

    -- The ''BoostingAttackTreeClassifier'' implemet boosting (SAMME) for
       classification problems.


"""

# Author: TanShouDong.



from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.externals import six
from sklearn.externals.six.moves import zip
from sklearn.externals.six.moves import xrange as range
from sklearn.ensemble.forest import BaseForest
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
from sklearn.metrics import accuracy_score, r2_score
from Utils.validation import check_array, check_X_y, check_random_state
from Utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.extmath import stable_cumsum



class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """
    Base class for BoostingAttachTree estimators.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod








