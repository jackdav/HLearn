import argparse
import numpy as np
import time

from models.tree import DecisionTreeClassifier
from utils.file_manager import load_data_split
from obj.candidate_set import CandidateDictionary

def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--root_dir', default='./data/', type=str)
    parser.add_argument('--params', required=False, type=str)
    return parser.parse_args()


def run():
    X_train, X_test, Y_train, Y_test = load_data_split(args.data, args.root_dir)

    begin = time.time()

    cd = CandidateDictionary(5)
    cd.generate_candidates(X_train)

    print(cd.set["6"].set)

    d_tree = DecisionTreeClassifier(features_idx=np.arange(1, X_train.shape[1]),
                                    max_depth=20, num_classes=2, candidate_dictionary=cd,
                                    hlearn=True, candidate_agreement_pctg=0.75)
    d_tree.fit(X_train, Y_train)
    d_tree.print_tree()
    print("Accuracy for novel solution: {}".format(d_tree.accuracy_score(X_test, Y_test)))

    end = time.time()

    # total time taken
    print("Runtime for novel solution: {}".format(end - begin))

    # This is for if you want to not use our solution

    begin = time.time()

    d_tree2 = DecisionTreeClassifier(features_idx=np.arange(1, X_train.shape[1]),
                                    max_depth=3, num_classes=2, candidate_dictionary=cd, hlearn=False)

    d_tree2.fit(X_train, Y_train)
    d_tree2.print_tree()
    print(d_tree2.root.feature)

    print("Accuracy for basic solution: {}".format(d_tree.accuracy_score(X_test, Y_test)))

    end = time.time()

    # total time taken
    print("Runtime for basic solution: {}".format(end - begin))


    return 0


if __name__== '__main__':
    args = load_args()
    run()