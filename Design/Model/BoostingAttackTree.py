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
    def __init__(self,
                 base_estimator,
                 n_estimators=50,
                 learning_rate=1.,
                 random_state=None):
        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,)
        self.learning_rate = learning_rate
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted fit method from the training set (X, y)
        :param X:
                {array-like, sparse matrix} of shape = [n_samples, n_features]
                The training input samples.Sparse matrix can be CSC, CSR, COO,
                DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        :param y:
                array-like of shape = [n_samples]
                The target values

        :param sample_weight:
                 array-like of shape = [n_samples], optional
                Sample weights. If None, the sample weights are initialized to
                1 / n_samples.

        :return:
                self : object

        """

        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))


        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]

        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()
        # Clear any previous fit results
        self.trained_estimators= []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        :param iboost: int
                The index of the current boost iteration.

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features]
                  The training input samples.

        :param y: array-like of shape = [n_samples]
                  The target values (class labels).

        :param sample_weight: array-like of shape = [n_samples]
                              The current sample weights.

        :param random_state: RandomState, The current random number generator.

        :return:
                sample_weight, estimator_weight, error.

        """
        pass


    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X




class BoostingAttackTreeClassifier(BaseWeightBoosting, ClassifierMixin):
    """
        An Trojan classifier
        Description:
                    这是一个分类迭代的过程，在每次迭代的过程中，相应比例的增大误分类样本的权重，
                    使得误分类的样本在后续迭代的过程中更加受攻击树分类器的注重，尽可能的对之前
                    误分类的样本分类正确。

        Parameters:
        base_estimator : object, optional，The base estimator from which the boosted ensemble is built.

         n_estimators : integer, optional (default=50)，The maximum number of estimators at which boosting is terminated.
                        In case of perfect fit, the learning procedure is stopped early.

        learning_rate : float, optional (default=1.),Learning rate shrinks the contribution of each classifier by
                      ``learning_rate``. There is a trade-off between ``learning_rate`` and``n_estimators``.

        random_state : int, RandomState instance or None, optional (default=None).

        Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    """

    def __init__(self,
                 base_estimator,
                 n_estimators=50,
                 learning_rate=1.,
                 # algorithm='SAMME.R',
                 algorithm='SAMME',
                 random_state=None):
        super(BoostingAttackTreeClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm=algorithm
        self.base_estimator=base_estimator


    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted BoostingAttackClassifier from the training set (X, y).

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features].

        :param y: array-like of shape = [n_samples],The target values (class labels).

        :param sample_weight:  array-like of shape = [n_samples], optional
                                Sample weights.If None, the sample weights are initialized to
                             ``1 / n_samples``.

        :return: self : object

        """

        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super(BoostingAttackTreeClassifier, self).fit(X, y, sample_weight)



    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        :param iboost: int
                The index of the current boost iteration.

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features]
                  The training input samples.

        :param y: array-like of shape = [n_samples]
                  The target values (class labels).

        :param sample_weight: array-like of shape = [n_samples]
                              The current sample weights.

        :param random_state: RandomState, The current random number generator.

        :return:
                sample_weight, estimator_weight, error.
        """
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight,
                                        random_state)


    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        # estimator = self._make_estimator(random_state=random_state)
        estimator = self.base_estimator

        estimator.fit(X, y, sample_weight=sample_weight)

        self.trained_estimators.append(estimator)

        y_predict_proba = estimator.predict_proba(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))

        #是否需要继续迭代
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error


    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        # estimator = self._make_estimator(random_state=random_state)
        estimator = self.base_estimator

        estimator.fit(X, y, sample_weight=sample_weight)

        self.trained_estimators.append(estimator)


        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.trained_estimators.pop(-1)
            if len(self.trained_estimators) == 0:
                raise ValueError('BaseClassifier in BoostingAttackTreeClassifier'
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
                np.log((1. - estimator_error) / estimator_error) +
                np.log(n_classes - 1.))

        # 是否需要继续迭代
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error


    def predict(self,X):
        """
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features]

        :return: y : array of shape = [n_samples],The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)


    def decision_function(self, X):
        """
        Compute the decision function of ``X``.

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features]

        :return:
        """

        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self.trained_estimators,
                                           self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred



































