import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class FeatureImportanceAttack:
    def __init__(self, X_train, y_train, feat_imp_method = "gini"):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.feat_imp_method = feat_imp_method
        self.fit()

        # This later can be used in feat_imp makes rank_method in function feat_imp redundant

    def fit(self):
        """
        Creates a dataframe of features in sorted order of their decreasing importance and their corresponding importances and attack direction
    
        """

        if self.feat_imp_method == "gini":
            scaler = (
                MinMaxScaler()
            )  # decision to use min-max is arbitrary - we can swap this as well
            self.scaler = scaler.fit(self.X_train)
            X_scaled = self.scaler.transform(self.X_train)
            feature_scores = (
                DecisionTreeClassifier()
                .fit(X_scaled, self.y_train)
                .feature_importances_
            )
        else:
            print(self.feat_imp_method," as feature importance method is not supported")
            exit()

        sorted_feature = pd.DataFrame(
            {"feature": self.X_train.columns, "importance": feature_scores},
        )
        X_scaled = pd.DataFrame(
            data=X_scaled, index=self.X_train.index, columns=self.X_train.columns
        )
        phish_mean = X_scaled.loc[self.y_train == 1].mean(axis=0)
        leg_mean = X_scaled.loc[self.y_train == 0].mean(axis=0)

        attack_direction = np.sign(leg_mean - phish_mean)  # returns a pd.Series
        sorted_feature["direction"] = attack_direction.values

        sorted_feature.sort_values(by=["importance"], ascending=False, inplace=True)

        # pandas dataframe with 3 columns, feature, importance, direction
        # sorted attacked direction
        self.sorted_attack_dir = sorted_feature

    def generate(self, X, y, e, n):
        """
        Creates adversarial samples from the dataset samples.

        Args:
            X (dataframe): Dataset to attack (every sample will be perturbed)
            y (dataframe): Y or label dataframe sample used by the attack
            e (float): Value between 0 and 1. Specifies how much a sample should be modified
            n (integer): Specifies the number of features to be modified


        Returns:
            dataframe : Contains the adversarial modified samples and legitimate samples which corresponds to the y label dataframe.
        """

        # creating a scaled data
        X_scaled = self.scaler.transform(X)
        # convert the X_scaled to dataframe
        X_scaled = pd.DataFrame(data=X_scaled, index=X.index, columns=X.columns)

        # list of features to be modified and their feature direction
        feature_mod = self.sorted_attack_dir["feature"][:n].to_list()
        feature_direction_value = self.sorted_attack_dir["direction"][:n].to_list()

        # epsilon budget
        real_e = X_scaled.loc[y == 1].sum(axis=1) * e
        e_budget = real_e / n
        phish_index_values = e_budget.index  # hack to get the phish indexes

        # type cast variables into numpy array and reshaping it for matrix multiplication
        e_budget_values = e_budget.values.reshape(-1, 1)
        feature_direction_value = np.array(feature_direction_value).reshape(1, -1)

        # Matrix Multiplication and adding the perturbations
        self.perturbation_matrix = e_budget_values @ feature_direction_value
        X_scaled.loc[phish_index_values, feature_mod] += self.perturbation_matrix

        # can be vectorized later
        for feature in feature_mod:
            X_scaled.loc[X_scaled[feature] < 0, feature] = 0

        # representing the data back in the original scale
        X_attack = self.scaler.inverse_transform(X_scaled)

        X_attack = pd.DataFrame(data=X_attack, index=X.index, columns=X.columns)

        # if direction is positive, round down, if negative round up.

        for feature_name in feature_mod:
            if (
                np.asscalar(
                    self.sorted_attack_dir[
                        self.sorted_attack_dir["feature"] == feature_name
                    ]["direction"].values
                )
                == 1
            ):
                X_attack[feature_name] = np.floor(X_attack[feature_name])
            else:
                X_attack[feature_name] = np.ceil(X_attack[feature_name])

        return X_attack

    def generate_random(self, X, y):
        """
        Creates samples which are generated using random values of e and n

        Args:
            X (dataframe): Dataset sample used by the model.
            y (dataframe): Y or label dataframe sample used by the model.
            scaled (bool, optional):  Scaling occurs if the dataset sample passed to it are not scaled. Users must set it
                                    to true if the dataset is already scaled. Defaults to False.

        Returns:
            dataframe : Contains the adversarial modified samples and legitimate samples which corresponds to the y label dataframe.
        """

        # creating a scaled data
        X_scaled = self.scaler.transform(X)
        # convert the X_scaled to dataframe
        X_scaled = pd.DataFrame(data=X_scaled, index=X.index, columns=X.columns)

        row_sum = X_scaled.loc[y == 1].sum(axis=1)
        row_sum_values = row_sum.values
        phish_index = row_sum.index

        n = [random.randint(1, 52) for _ in range(len(phish_index))]

        e = np.random.uniform(0.0, 1.0, len(phish_index))

        # calculating the perturbation per feature
        total_perturbation = [a * b for a, b in zip(row_sum_values, e)]
        perturbation_per_feature = [a / b for a, b in zip(total_perturbation, n)]

        # Initializing empty perturbation matrix as an empty dataframe
        self.perturbation_matrix = pd.DataFrame(
            0.00, index=np.arange(len(phish_index)), columns=X.columns
        )

        # populating the perturbation matrix
        for i in range(len(self.perturbation_matrix)):
            self.perturbation_matrix.loc[
                i, self.sorted_attack_dir["feature"][: n[i]].to_list()
            ] += (
                perturbation_per_feature[i]
                * self.sorted_attack_dir["direction"][: n[i]]
            ).to_list()

        # adding the perturbation to X_scaled
        X_scaled.loc[phish_index] += self.perturbation_matrix.to_numpy()

        X_scaled[X_scaled < 0] = 0

        X_attack = self.scaler.inverse_transform(X_scaled)

        X_attack = pd.DataFrame(data=X_attack, index=X.index, columns=X.columns)

        for feature_name in X.columns:
            if (
                np.asscalar(
                    self.sorted_attack_dir[
                        self.sorted_attack_dir["feature"] == feature_name
                    ]["direction"].values
                )
                == 1
            ):
                X_attack[feature_name] = np.floor(X_attack[feature_name])
            else:
                X_attack[feature_name] = np.ceil(X_attack[feature_name])

        return X_attack

