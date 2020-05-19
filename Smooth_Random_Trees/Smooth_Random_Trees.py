''' A class that builds Differentially Private Random Decision Trees,
    using Smooth Sensitivity
    by Sam Fletcher
    www.samfletcher.work

    Refactored by Ben Zhao to expoze commonly used ML api
    functions like fit, predict. Multi-processing has been removed.
'''

####################################################################
from collections import Counter, defaultdict
import random
import numpy as np
import math
import warnings
# local module
import node


class DP_Random_Forest:
    '''Make a forest of Random Trees, then filter the training
       data through each tree to fill the leafs.
       IMPORTANT: the first attribute of both the training and
       testing data MUST be the class attribute. '''

    def __init__(self, epsilon, n_labels, num_trees=100, MULTI_THREAD=False, pool_size=None):

        '''Some initialization'''
        self.epsilon = epsilon
        # All datasets are nuemrical, or have been previously binarized to numerals
        self._categs = []
        self._num_trees = num_trees
        self._nlabels = n_labels
        self.MULTI_THREAD = MULTI_THREAD
        if MULTI_THREAD:
            assert pool_size is not None
        self.pool_size = pool_size
        # disabling multithreading on predictions
        self.MULTI_THREAD_PREDICT = False

    def fit(self,
            train,
            labels=None,
            split_labels=True,
            categs=[],
            MULTI_PROCESSING=False
            ):
        '''Some initialization'''
        if split_labels:
            assert labels is not None, 'No split labels provided.'
            assert len(train) == len(labels), 'Data and Labels are not the same size'
            new_train = np.c_[labels, train]
            train = new_train.tolist()

        self._categs = categs
        # The indexes of the numerical (i.e. continuous) attributes
        # Numeral features
        numers = [x+1 for x in range(len(train[0])-1) if x+1 not in categs]
        self._attribute_domains = self.get_attr_domains(train, numers, categs)
        attribute_indexes = [int(k) for k, v in self._attribute_domains.items()]
        self._max_depth = self.calc_tree_depth(len(numers), len(categs))
        self._n_features = len(train[0]) - 1

        random.shuffle(train)
        # domain of labels
        
        #print(train.shape)
        print(type(train))
        #print(len(train.tolist()[0]))
        
        class_labels = list(set([int(x[0]) for x in train]))

        # by using subsets, we don't need to divide the epsilon budget
        subset_size = int(len(train)/self._num_trees)
        
        if self.MULTI_THREAD:
            from functools import partial
            from multiprocessing import Pool
            variables = [train[i*subset_size:(i+1)*subset_size] for i in range(self._num_trees)]
            func = partial(self.build_tree, attribute_indexes=attribute_indexes, epsilon=self.epsilon, class_labels=class_labels, test=[])
            p = Pool(self.pool_size)
            result_holder = p.map(func, variables)
            tree_holder = [res['tree'] for res in result_holder]
            p.close()
        else:
            tree_holder = []
            for i in range(self._num_trees):
                results = self.build_tree(
                    train[i*subset_size:(i+1)*subset_size],
                    attribute_indexes, self.epsilon, class_labels, []
                )
                tree_holder.append(results['tree'])
        self._trees = tree_holder

    def _predict_proba_vec(self, x_vec):
        predict_vec = []
        for i in range(self._num_trees):
            tree = self._trees[i]
            result = tree.classify(x_vec, tree._root_node)
            predict_vec.append(result)

        # convert predicted labels to a confidence array of all labels
        _unique, _count = np.unique(predict_vec, return_counts=True)
        _temp = dict(zip(_unique, _count))
        _list = [_temp.get(_t, 0) for _t in range(self._nlabels)]
        assert sum(_count) == self._num_trees
        predict_vec = np.array(_list)/sum(_count)
        return predict_vec
        
    def predict_proba(self, x):
        if len(x[0]) == self._n_features + 1:
            warnings.warn('Assuming first column is the class labels')
        else:
            x = np.c_[np.zeros(len(x)), np.array(x)].tolist()
        assert len(x[0]) == (self._n_features + 1), 'Incorrect feature size'
        
        if self.MULTI_THREAD_PREDICT:
            from functools import partial
            from multiprocessing import Pool
            variables = x
            p = Pool(self.pool_size)
            # map preserves the order of the input variables.
            predictions = p.map(self._predict_proba_vec, variables)
            p.close()
            
        else:
            predictions = []
            for x_vec in x:
                predict_vec = self._predict_proba_vec(x_vec)
                predictions.append(predict_vec)
            
#             for x_vec in x:
#                 print("new_vector")
#                 predict_vec = []
#                 for i in range(self._num_trees):
#     #                  # Attempt at creating probabalistic tree from leaf purity.
#     #                 tree = self._trees[i]
#     #                 result = tree.classify(x_vec, tree._root_node, all_counts=True)
#     #                 label, counts = result
#     #                 total = sum(counts.values())
#     #                 for k in range(self._nlabels):
#     #                     predict_vec.append(1.0*counts.get(k, 0)/total)

#                     tree = self._trees[i]
#                     result = tree.classify(x_vec, tree._root_node)
#                     predict_vec.append(result)

#                 # convert predicted labels to a confidence array of all labels
#                 _unique, _count = np.unique(predict_vec, return_counts=True)
#                 _temp = dict(zip(_unique, _count))
#                 _list = [_temp.get(_t, 0) for _t in range(self._nlabels)]
#                 assert sum(_count) == self._num_trees
#                 predict_vec = np.array(_list)/sum(_count)
#                 predictions.append(predict_vec)

        return np.array(predictions)

    def _predict_vec(self, x_vec):
        predict_vec = []
        for j in range(self._num_trees):
            tree = self._trees[j]
            result = tree.classify(x_vec, tree._root_node)
            predict_vec.append(result)

        assert len(predict_vec) == self._num_trees
        predict_y = Counter(predict_vec).most_common(1)[0][0]
        return predict_y

    def predict(self, x):

        if len(x[0]) == self._n_features + 1:
            warnings.warn('Assuming first column is the class labels')
        else:
            x = np.c_[np.zeros(len(x)), np.array(x)].tolist()
        assert len(x[0]) == (self._n_features + 1), 'Incorrect feature size'
        
        if self.MULTI_THREAD_PREDICT:
            from functools import partial
            from multiprocessing import Pool
            variables = x
            p = Pool(self.pool_size)
            # map preserves the order of the input variables.
            predictions = p.map(self._predict_vec, variables)
            p.close()
            
        else:
            predictions = []
            for x_vec in x:
                predictions.append(self._predict_vec(x_vec))
        
#             predictions = []
#             for x_vec in x:
#                 predict_vec = []
#                 for j in range(self._num_trees):
#                     tree = self._trees[j]
#                     result = tree.classify(x_vec, tree._root_node)
#                     predict_vec.append(result)

#                 assert len(predict_vec) == self._num_trees
#                 predict_y = Counter(predict_vec).most_common(1)[0][0]
#                 predictions.append(predict_y)

        return predictions

    def complete_test_v2(self, train, test, categs, num_trees, epsilon):
        '''
            train, 2D list of the training data where the columns
                   are the attributes, and the first column are
                   the class attributes.
            test, 2D list of the testing data where the columns
                  are the attributes, and the first column is
                  the class attribute.
            categs, The indexes of the categorical (i.e. discrete)
                    attributes, EXCLUDING the class attribute.
            num_trees, The number of trees to build. Since we divide
                       the data among the trees, ensure:
                       num_trees << len(train)
            epsilon, The total privacy budget. The whole budget will
                     be used in each tree (and thus each leaf),
                     due to using disjoint data.
        '''
        '''Some initialization'''
        self._categs = categs
        # The indexes of the numerical (i.e. continuous) attributes
        numers = [x+1 for x in range(len(train[0])-1) if x+1 not in categs]
        self._attribute_domains = self.get_attr_domains(train, numers, categs)
        attribute_indexes = [int(k) for k, v in self._attribute_domains.items()]
        self._max_depth = self.calc_tree_depth(len(numers), len(categs))
        self._num_trees = num_trees

        '''Some bonus information gained throughout the algorithm'''
        self._missed_records = []
        self._flipped_majorities = []
        self._av_sensitivity = []
        self._empty_leafs = []

        random.shuffle(train)
        # domain of labels
        class_labels = list(set([int(x[0]) for x in train]))
        # ordered list of the test data labels
        actual_labels = [int(x[0]) for x in test]
        # initialize
        voted_labels = [defaultdict(int) for x in test]

        # by using subsets, we don't need to divide the epsilon budget
        subset_size = int(len(train)/self._num_trees)

        debug_holder = []

        for i in range(self._num_trees):
            results = self.build_tree(
                train[i*subset_size:(i+1)*subset_size],
                attribute_indexes, epsilon, class_labels, test
            )
            debug_holder.append(results)

            '''Collect the predictions and the bonus information'''
            curr_votes = results['voted_labels']
            for rec_index in range(len(test)):
                for lab in class_labels:
                    voted_labels[rec_index][lab] += curr_votes[rec_index][lab]
            self._missed_records.append(results['missed_records'])
            self._flipped_majorities.append(results['flipped_majorities'])
            self._av_sensitivity.append(results['av_sensitivity'])
            self._empty_leafs.append(results['empty_leafs'])

        final_predictions = []
        for i, rec in enumerate(test):
            final_predictions.append(Counter(voted_labels[i]).most_common(1)[0][0])
        counts = Counter([x == y for x, y in zip(final_predictions, actual_labels)])
        self._predicted_labels = final_predictions
        self._accuracy = float(counts[True]) / len(test)

        return debug_holder

    def get_attr_domains(self, data, numers, categs):
        attr_domains = {}
        transData = np.transpose(data)
        for i in categs:
            attr_domains[str(i)] = [str(x) for x in set(transData[i])]
            print("original domain length of categ att {}: {}".format(i, len(attr_domains[str(i)])))
        for i in numers:
            vals = [float(x) for x in transData[i]]
            attr_domains[str(i)] = [min(vals), max(vals)]
        return attr_domains

    def calc_tree_depth(self, num_numers, num_categs):
        # if no numerical attributes
        if num_numers < 1:
            # depth = half the number of categorical attributes
            return math.floor(num_categs/2.)
        else:
            ''' Designed using balls-in-bins probability. See the paper for details. '''
            m = float(num_numers)
            depth = 0
            # the number of unique attributes not selected so far
            expected_empty = m
            # repeat until we have less than half the attributes being empty
            while expected_empty > m/2.:
                expected_empty = m * ((m-1.)/m)**depth
                depth += 1
            # The above was only for half the numerical attributes.
            # Now add half the categorical attributes
            final_depth = math.floor(depth + (num_categs/2.))
            ''' WARNING: The depth translates to an exponential increase in memory usage.
                Do not go above ~15 unless you have 50+ GB of RAM. '''
            return min(15, final_depth)

    def build_tree(self, train, attribute_indexes, epsilon, class_labels, test):
        root = random.choice(attribute_indexes)
        tree = Tree(attribute_indexes, root, self)
        tree.filter_training_data_and_count(train, epsilon, class_labels)

        missed_records = tree._missed_records
        flipped_majorities = tree._flip_fraction
        av_sensitivity = tree._av_sensitivity
        empty_leafs = tree._empty_leafs
        voted_labels = [defaultdict(int) for x in test]

        for i, rec in enumerate(test):
            label = tree.classify(rec, tree._root_node)
            voted_labels[i][label] += 1

        return {'voted_labels': voted_labels,
                'missed_records': missed_records,
                'flipped_majorities': flipped_majorities,
                'av_sensitivity': av_sensitivity,
                'empty_leafs': empty_leafs,
                'tree': tree}


class Tree(DP_Random_Forest):
    ''' Set the root for this tree and then start the random-tree-building process. '''
    def __init__(self, attribute_indexes, root_attribute, pc):
        self._id = 0
        self._categs = pc._categs
        self._max_depth = pc._max_depth
        self._num_leafs = 0

        root = node.node(None, None, root_attribute, 1, 0, [])  # the root node is level 1
        attribute_domains = pc._attribute_domains

        # Numerical attribute
        if root_attribute not in self._categs:
            split_val = random.uniform(attribute_domains[str(root_attribute)][0],
                                       attribute_domains[str(root_attribute)][1])
            left_domain = {k: v if k != str(root_attribute) else
                           [v[0], split_val] for k, v in attribute_domains.items()}
            right_domain = {k: v if k != str(root_attribute) else
                            [split_val, v[1]] for k, v in attribute_domains.items()}
            # add left child
            root.add_child(self.make_children([x for x in attribute_indexes],
                                              root, 2, '<' + str(split_val),
                                              split_val, left_domain))
            # add right child
            root.add_child(self.make_children([x for x in attribute_indexes],
                                              root, 2, '>=' + str(split_val),
                                              split_val, right_domain))
        # Categorical attribute
        else:
            assert False
            for value in attribute_domains[str(root_attribute)]:
                root.add_child(
                    # categorical attributes can't be tested again
                    self.make_children([x for x in attribute_indexes if x != root_attribute],
                                       root, 2, value, None, attribute_domains))
        self._root_node = root

    ''' Recursively make all the child nodes for the current node,
        until a termination condition is met.'''
    def make_children(self,
                      candidate_atts,
                      parent_node,
                      current_depth,
                      splitting_value_from_parent,
                      svfp_numer,
                      attribute_domains):
        self._id += 1
        # termination conditions. leaf nodes don't count to the depth.
        if not candidate_atts or current_depth >= self._max_depth+1:
            self._num_leafs += 1
            return node.node(parent_node,
                             splitting_value_from_parent,
                             None,
                             current_depth,
                             self._id,
                             None,
                             svfp_numer=svfp_numer)
        else:
            # pick the attribute that this node will split on
            new_splitting_attr = random.choice(candidate_atts)
            # make a new node
            current_node = node.node(parent_node,
                                     splitting_value_from_parent,
                                     new_splitting_attr,
                                     current_depth,
                                     self._id,
                                     [],
                                     svfp_numer=svfp_numer)

            # Numerical attribute
            if new_splitting_attr not in self._categs:
                split_val = random.uniform(attribute_domains[str(new_splitting_attr)][0],
                                           attribute_domains[str(new_splitting_attr)][1])
                left_domain = {k: v if k != str(new_splitting_attr) else
                               [v[0], split_val] for k, v in attribute_domains.items()}
                right_domain = {k: v if k != str(new_splitting_attr) else
                                [split_val, v[1]] for k, v in attribute_domains.items()}
                # Left child
                current_node.add_child(self.make_children(
                    [x for x in candidate_atts],
                    current_node,
                    current_depth+1,
                    '<',
                    split_val,
                    left_domain))
                # Right child
                current_node.add_child(self.make_children(
                    [x for x in candidate_atts],
                    current_node,
                    current_depth+1,
                    '>=',
                    split_val,
                    right_domain))
            # Categorical attribute
            else:
                # for every value in the splitting attribute
                for value in attribute_domains[str(new_splitting_attr)]:
                    child_node = self.make_children(
                        [x for x in candidate_atts if x != new_splitting_attr],
                        current_node, current_depth+1, value, None, attribute_domains)
                    # add children to the new node
                    current_node.add_child(child_node)
            return current_node

    ''' Record which leaf each training record belongs to,
        and then set the (noisy) majority label. '''
    def filter_training_data_and_count(self, records, epsilon, class_values):
        ''' epsilon = the epsilon budget for this tree
            (each leaf is disjoint, so the budget can be re-used). '''
        num_unclassified = 0.

        for rec in records:
            num_unclassified += self.filter_record(rec, self._root_node, class_index=0)
        self._missed_records = num_unclassified
        _temp = self.set_all_noisy_majorities(epsilon, self._root_node, class_values, 0, 0, [])
        flipped_majorities, empty_leafs, sensitivities = _temp

        # excludes empty leafs
        self._av_sensitivity = np.mean(sensitivities)

        if self._num_leafs == 0:
            print("\n\n~~~ WARNING: NO LEAFS. num_unclassified = " + str(num_unclassified) + " ~~~\n\n")
            self._empty_leafs = -1.0
        else:
            self._empty_leafs = empty_leafs / float(self._num_leafs)

        if empty_leafs == self._num_leafs:
            print("\n\n~~~ WARNING: all leafs are empty. num_unclassified = " + str(num_unclassified) + " ~~~\n\n")
            self._flip_fraction = -1.0
        else:
            self._flip_fraction = flipped_majorities / float(self._num_leafs-empty_leafs)

    def filter_record(self, record, node, class_index=0):
        # For debugging purposes. Doesn't happen in my experience
        if not node:
            assert False, "Error 0.00001"
            return 0.00001
        # if leaf
        if not node._children:
            node.increment_class_count(record[class_index])
            return 0.
        else:
            child = None
            # numerical attribute
            if node._splitting_attribute not in self._categs:
                rec_val = record[node._splitting_attribute]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            # categorical attribute
            else:
                rec_val = str(record[node._splitting_attribute])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            # if the record's value couldn't be found:
            if child is None and node._splitting_attribute in self._categs:
                # For debugging purposes
                assert False, "Error 1."
                return 1.
            # if the record's value couldn't be found
            elif child is None:
                # For debugging purposes
                assert False, "Error 0.001"
                return 0.001
            return self.filter_record(record, child, class_index)

    def set_all_noisy_majorities(self,
                                 epsilon,
                                 node,
                                 class_values,
                                 flipped_majorities,
                                 empty_leafs,
                                 sensitivities):
        if node._children:
            for child in node._children:
                flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(
                    epsilon, child, class_values, flipped_majorities, empty_leafs, sensitivities)
        else:
            flipped_majorities += node.set_noisy_majority(epsilon, class_values)
            empty_leafs += node._empty
            if node._sensitivity >= 0.0:
                sensitivities.append(node._sensitivity)
        return flipped_majorities, empty_leafs, sensitivities

    def classify(self, record, node, all_counts=False):
        if not node:
            return None
        elif not node._children:  # if leaf
            if not all_counts:
                return node._noisy_majority
            else:
                print(node._noisy_majority, node._class_counts)
                return node._noisy_majority, node._class_counts
        else:  # if parent
            attr = node._splitting_attribute
            child = None
            if node._splitting_attribute not in self._categs:  # numerical attribute
                rec_val = record[attr]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else:  # categorical attribute
                assert False
                rec_val = str(record[attr])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            # if the record's value couldn't be found,
            # just return the latest majority value
            if child is None:
                # return majority_value, majority_fraction
                if not all_counts:
                    return node._noisy_majority
                else:
                    print(node._noisy_majority, node._class_counts)
                    return node._noisy_majority, node._class_counts
            else:
                return self.classify(record, child, all_counts=all_counts)


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_wine
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    def api_test(train_xy, test_xy):
        y = np.array(train_xy, dtype=int)[:, 0].tolist()
        x = np.array(train_xy)[:, 1:].tolist()

        forest = DP_Random_Forest(num_trees=n_trees, epsilon=epsilon, n_labels=len(set(y)))

        forest.fit(x, y)
        test = np.array(test_xy)[:, 1:].tolist()
        predict_y = forest.predict(test)

        acc = accuracy_score(list(np.array(test_xy, dtype=int)[:, 0]), list(np.array(predict_y, dtype=int)))
        del forest
        return acc

    def base_test(train_xy, test_xy):
        forest = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
        y = np.array(train_xy, dtype=int)[:, 0].tolist()
        x = np.array(train_xy)[:, 1:].tolist()

        forest.fit(x, y)
        test = np.array(test_xy)[:, 1:].tolist()
        predict_y = forest.predict(test)

        acc = accuracy_score(list(np.array(test_xy, dtype=int)[:, 0]), list(np.array(predict_y, dtype=int)))
        del forest
        return acc

    def complete_test(train_xy, test_xy):
        y = np.array(train_xy, dtype=int)[:, 0].tolist()

        forest = DP_Random_Forest(epsilon=None, n_labels=len(set(y)))
        forest.complete_test_v2(train_xy, test_xy, [], n_trees, epsilon)
        acc = forest._accuracy
        del forest
        return acc

    # Begin Implementation testing function
    data = load_wine()

    print('Data: {}'.format(data.data.shape))
    print('Labels: {}'.format(data.target.shape))

    labelled_data = np.c_[data.target, data.data]
    a = train_test_split(labelled_data, test_size=0.2)
    train_xy, test_xy = a

    epsilon = 500
    n_trees = 5
    max_depth = len(set(data.target))
    print('epsilon: {}'.format(epsilon))
    print('n_trees: {}'.format(n_trees))
    print('max_depth: {}'.format(max_depth))

    # Verify implementation for api config
    print('API')
    api_holder = []
    for _ in tqdm(range(100)):
        a = train_test_split(labelled_data, test_size=0.2)
        train_xy, test_xy = a
        api_holder.append(api_test(train_xy, test_xy))

    # Verify implementation for a base rndf classifier
    print('Regular RNDF')
    base_holder = []
    for _ in tqdm(range(100)):
        a = train_test_split(labelled_data, test_size=0.2)
        train_xy, test_xy = a
        base_holder.append(base_test(train_xy, test_xy))

    # obtain previous monolithic implementation baseline
    print('Complete')
    complete_holder = []
    for _ in tqdm(range(100)):
        a = train_test_split(labelled_data, test_size=0.2)
        train_xy, test_xy = a
        complete_holder.append(complete_test(train_xy, test_xy))

    # Ensure outputs of api results, and complete results are simmilar.
    print('API_ver: {}'.format(np.mean(api_holder)))
    print('RNDF_basic: {}'.format(np.mean(base_holder)))
    print('Complete_ver: {}'.format(np.mean(complete_holder)))


# ''' A toy example of how to call the class '''
# if __name__ == '__main__':
#     data = [[1, 'a', 12, 3, 14],
#             [1, 'a', 12, 3, 4],
#             [1, 'a', 12, 3, 4],
#             [0, 'a', 2, 13, 4],
#             [0, 'b', 2, 13, 14],
#             [0, 'b', 2, 3, 14],
#             ]

#     forest = DP_Random_Forest(data[1:], data, [1, ], 2, 0.1)
#     print('accuracy = ' + str(forest._accuracy))
