import numpy as np
from sklearn.utils import resample
from sklearn.base import clone


class MPD:
    def __init__(self, X_train, y_train, base_clf, num_bootstrap=100):
        """
        :param bootstrap: number of samples to bootstrap.
        :param X_train: The training data to use for your MPD detector
        :param y_train: A vector of labels for the training data
        :base_clf: any sklearn estimator
        :num_bootstrap : number of times you want to bootstrap

        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_bootstrap = num_bootstrap
        self.bootstrap_clfs = []
        self.base_clf = base_clf
        self.fit()

    def fit(self):
        """        
        This function resamples the X and y data and trains self.num_bootstrap models on them.
        It does not return anything, it sets self.bootstrap_clfs

        """

        for _ in range(self.num_bootstrap):
            X, y = resample(self.X_train, self.y_train, replace=True)
            clf = clone(self.base_clf)
            clf.fit(X, y)
            self.bootstrap_clfs.append(clf)

    def get_mpd_score(self, X):
        """
        :param X: An array of features samples to measure the MPD on
        
        :return: array of mpd values for the input array
        """

        total_probs = []
        for clf in self.bootstrap_clfs:  #
            probs_classes = clf.predict_proba(X)
            # this returns number of rows x number of classes
            probs = probs_classes[:, 1]  # probability of 1th class (malicious)
            total_probs.append(probs)
        self.total_probs_ = np.array(total_probs)

        # mpd.......

        U_0_X = (self.total_probs_ - 0) ** 2
        U_0_X = U_0_X.sum(axis=0)
        U_0_X = np.sqrt(U_0_X / self.num_bootstrap)

        U_1_X = (self.total_probs_ - 1) ** 2
        U_1_X = U_1_X.sum(axis=0)
        U_1_X = np.sqrt(U_1_X / self.num_bootstrap)

        mpd = np.minimum(U_0_X, U_1_X)
        return mpd

