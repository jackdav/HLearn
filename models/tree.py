import numpy as np

class Node:
    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier:

    def __init__(self, max_depth=None, features_idx=None, num_classes=2, candidate_dictionary=None,
                 candidate_agreement_flag=False, candidate_agreement_pctg=1.0):
        self.features_idx = features_idx
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.cd = candidate_dictionary
        self.candidate_agreement_flag = candidate_agreement_flag
        self.candidate_agreement_pctg = candidate_agreement_pctg

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]
        return preds

    def print_tree(self):
        q = [self.root]
        while q:
            niq = q[0]
            q.pop(0)
            if niq.left_tree is not None:
                q.append(niq.left_tree)
            if niq.right_tree is not None:
                q.append(niq.right_tree)
            print("Feature: {} | Split: {} ||".format(niq.feature, niq.split))

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


    # gets data corresponding to a split by using numpy indexing
    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] >= split)
        right_idx = np.where(X[:, feature] < split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]
        # calculate gini impurity and gain for y, left_y, right_y
        if self.candidate_agreement_flag:
            agreement_count = len(y)

            for i in left_idx[0]:
                for j in self.cd.set[str(i + 1)].set:
                    if j[feature] < split:
                        agreement_count -= 1

            for i in right_idx[0]:
                for j in self.cd.set[str(i + 1)].set:
                    if j[feature] >= split:
                        agreement_count -= 1

            pctg = agreement_count / len(y)
            if pctg >= self.candidate_agreement_pctg:
                gain = self.calculate_gini_gain(y, left_y, right_y)
                return gain, left_X, right_X, left_y, right_y
            else:
                return 0.0, left_X, right_X, left_y, right_y
        else:
            gain = self.calculate_gini_gain(y, left_y, right_y)
            return gain, left_X, right_X, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        if len(left_y) > 0 and len(right_y) > 0:
            denom = np.count_nonzero(y==1) + np.count_nonzero(y==0)
            ua = 1 - (np.count_nonzero(y==1)/denom)**2 - (np.count_nonzero(y==0)/denom)**2
            lp = np.count_nonzero(left_y==1)
            ln = np.count_nonzero(left_y==0)
            ual = 1 - (lp/(ln+lp))**2 - (ln/(ln+lp))**2
            pl = (ln+lp)/denom
            rp = np.count_nonzero(right_y == 1)
            rn = np.count_nonzero(right_y == 0)
            uar = 1 - (rp / (rn + rp)) ** 2 - (rn / (rn + rp)) ** 2
            pr = (rn + rp) / denom
            gain = ua - (pl*ual) - (pr*uar)
            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0
