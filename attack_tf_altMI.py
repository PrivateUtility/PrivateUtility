from classifier_tf import train as train_model, load_dataset, get_predictions
from classifier_tf import train_classic as train_classic_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from scipy import stats
import numpy as np
import tensorflow as tf
import argparse
import os
import imp
import pickle
import matplotlib.pyplot as plt
import random
import warnings

from resource_tracking import resource_tracking


MODEL_PATH = './model/'
DATA_PATH = './data/'
RESULT_PATH = './zhaoai_results/'

SMALL_VALUE = 0.001


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def get_data_indices(data_size, target_train_size=int(1e4), sample_target_data=True):
    train_indices = np.arange(data_size)
    if sample_target_data:
        target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
        shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    else:
        target_data_indices = train_indices[:target_train_size]
        shadow_indices = train_indices[target_train_size:]
    return target_data_indices, shadow_indices


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_classic_target_model(dataset, l2_ratio=1e-7, model='nn', save=True,
                               privacy='no_privacy', preprocess=None, dp='dp', epsilon=0.5, delta=1e-5, resource_tracker=None):

    train_x, train_y, test_x, test_y, n_classes = dataset

    a = train_classic_model(dataset, model=model, silent=False,
                            privacy=privacy, preprocess=preprocess, dp=dp, epsilon=epsilon, delta=delta,         
                            resource_tracker=resource_tracker)
    classifier, _, _, train_loss, train_acc, test_acc = a

    # test data for attack model
    attack_x, attack_y = [], []

    # data used in training, label is 1
    pred_scores = classifier.predict_proba(train_x)
    nan_vals = np.isnan(pred_scores)
    if nan_vals.any():
        warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
        pred_scores[nan_vals] = 1.0/n_classes
    attack_x.append(pred_scores)
    attack_y.append(np.ones(train_x.shape[0]))

    # data not used in training, label is 0
    pred_scores = classifier.predict_proba(test_x)
    nan_vals = np.isnan(pred_scores)
    if nan_vals.any():
        warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
        pred_scores[nan_vals] = 1.0/n_classes
    attack_x.append(pred_scores)
    attack_y.append(np.zeros(test_x.shape[0]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    assert save == False, "Defunct option"

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes, train_loss, classifier, train_acc, test_acc


def train_target_model(dataset, epochs=100, batch_size=100, learning_rate=0.01,
                       l2_ratio=1e-7, n_hidden=50, model='nn', save=True,
                       privacy='no_privacy', preprocess=None, dp='dp', epsilon=0.5, delta=1e-5, resource_tracker=None):

    train_x, train_y, test_x, test_y, n_classes = dataset

    a = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                    batch_size=batch_size, model=model, l2_ratio=l2_ratio, silent=False, 
                    privacy=privacy, preprocess=preprocess, dp=dp, epsilon=epsilon, delta=delta,        
                    resource_tracker=resource_tracker)
    classifier, _, _, train_loss, train_acc, test_acc = a

    # test data for attack model
    attack_x, attack_y = [], []

    # data used in training, label is 1
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)

    attack_x.append(pred_scores)
    attack_y.append(np.ones(train_x.shape[0]))

    # data not used in training, label is 0
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)

    attack_x.append(pred_scores)
    attack_y.append(np.zeros(test_x.shape[0]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    assert save == False, "Defunct option"

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
        np.savez(MODEL_PATH + 'target_model.npz', *lasagne.layers.get_all_param_values(output_layer))

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes, train_loss, classifier, train_acc, test_acc


def sample_data_anew(args):
    print('-' * 10 + 'LOADING DATA DIRECT TO EXPERIMENT' + '-' * 10 + '\n')

    ####################################
    # Load the correct source datafile
    if 'purchase' in args.train_dataset:
        x = pickle.load(open('dataset/purchase_features.p', 'rb'))
    elif 'netflix' in args.train_dataset:
        x = pickle.load(open('dataset/netflix_features.p', 'rb'))
    elif 'cifar' in args.train_dataset:
        x = pickle.load(open('dataset/cifar_features.p', 'rb'))
    elif 'synthetic' in args.train_dataset:
        x = pickle.load(open('dataset/synthetic_features.p', 'rb'))
    else:
        x = pickle.load(open('dataset/'+args.train_dataset+'_features.p', 'rb'))
    y = pickle.load(open('dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x, y = np.matrix(x), np.array(y)

    ####################################
    # Partition the data
    test_x, test_y = None, None
    y = y.astype('int32')
    x = x.astype(np.float32)
    if test_x is None:
        print('Splitting train/test data with ratio {}/{}'.format(1 - args.test_ratio, args.test_ratio))
        x, test_x, y, test_y = train_test_split(x, y, test_size=args.test_ratio, stratify=y)

    # need to partition target and shadow model data
    assert len(x) > 2 * args.target_data_size

    target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=args.target_data_size)

    # target model's data
    print('Saving data for target model')
    train_x, train_y = x[target_data_indices], y[target_data_indices]
    size = len(target_data_indices)
    if size < len(test_x):
        test_x = test_x[:size]
        test_y = test_y[:size]

    # If a label does not appear in either train or test, it does not exist.
    n_classes = len(np.unique(np.r_[train_y, test_y]))

    return train_x, train_y, test_x, test_y, n_classes


def save_data(args):
    print('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')

    ####################################
    # Load the correct source datafile
    if 'purchase' in args.train_dataset:
        x = pickle.load(open('dataset/purchase_features.p', 'rb'))
    elif 'netflix' in args.train_dataset:
        x = pickle.load(open('dataset/netflix_features.p', 'rb'))
    else:
        x = pickle.load(open('dataset/'+args.train_dataset+'_features.p', 'rb'))
    y = pickle.load(open('dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x, y = np.matrix(x), np.array(y)

    ####################################
    # set the data saving path
    NESTED_PATH = '{}/'.format(args.train_dataset)

    ####################################
    # Partition the data
    test_x, test_y = None, None
    y = y.astype('int32')
    if test_x is None:
        print('Splitting train/test data with ratio {}/{}'.format(1 - args.test_ratio, args.test_ratio))
        x, test_x, y, test_y = train_test_split(x, y, test_size=args.test_ratio, stratify=y)

    # need to partition target and shadow model data
    assert len(x) > 2 * args.target_data_size

    target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=args.target_data_size)
    os.makedirs(MODEL_PATH + NESTED_PATH, exist_ok=True)
    np.savez(MODEL_PATH + NESTED_PATH + 'data_indices.npz', target_data_indices, shadow_indices)
    # target model's data
    print('Saving data for target model')
    train_x, train_y = x[target_data_indices], y[target_data_indices]
    size = len(target_data_indices)
    if size < len(test_x):
        test_x = test_x[:size]
        test_y = test_y[:size]

    ####################################
    # save target data
    os.makedirs(DATA_PATH + NESTED_PATH, exist_ok=True)
    np.savez(DATA_PATH + NESTED_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    if not args.ignore_shadow:
        # shadow model's data
        target_size = len(target_data_indices)
        shadow_x, shadow_y = x[shadow_indices], y[shadow_indices]
        shadow_indices = np.arange(len(shadow_indices))

        for i in range(args.n_shadow):
            print('Saving data for shadow model {}'.format(i))
            shadow_i_indices = np.random.choice(shadow_indices, 2 * target_size, replace=False)
            shadow_i_x, shadow_i_y = shadow_x[shadow_i_indices], shadow_y[shadow_i_indices]
            train_x, train_y = shadow_i_x[:target_size], shadow_i_y[:target_size]
            test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
            np.savez(DATA_PATH + NESTED_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    # If a label does not appear in either train or test, it does not exist.
    n_classes = len(np.unique(np.r_[train_y, test_y]))

    return train_x, train_y, test_x, test_y, n_classes


def membership_inference(true_y, pred_y, membership, train_loss):
    print('-' * 10 + 'MEMBERSHIP INFERENCE' + '-' * 10 + '\n')
    pred_membership = np.where(log_loss(true_y, pred_y) <= train_loss, 1, 0)
    fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
    print(fpr, tpr, tpr-fpr)
    mem_adv = tpr[1]-fpr[1]
    return mem_adv, log_loss(true_y, pred_y)


def mlleak_membership_inference(pred_y, membership):
    print('-' * 10 + 'ML-LEAKS MEMBERSHIP INFERENCE' + '-' * 10 + '\n')
    fpr, tpr, thresholds = roc_curve(list(membership), np.squeeze(np.asarray(np.max(pred_y, axis=1))), pos_label=1)
    membership_auc = auc(fpr, tpr)
    print("AUC: {}".format(membership_auc))
    return membership_auc


def attribute_inference(true_x, true_y, batch_size, classifier, train_loss, features, n_classes=None, classic_model=False):
    # Batch Size is a redundant input
    print('-' * 10 + 'ATTRIBUTE INFERENCE' + '-' * 10 + '\n')

    if classic_model:
        assert n_classes is not None, "Please specify number of classes for classic models."

    attr_adv, attr_mem, attr_pred = [], [], []
    for feature in features:
        low_op, high_op = [], []

        low_data, high_data, membership = getAttributeVariations(true_x, feature)

        print('Variations Gened {}'.format(feature))

        if not classic_model:
            pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': low_data},
                num_epochs=1,
                shuffle=False)

            predictions = classifier.predict(input_fn=pred_input_fn)
            _, low_op = get_predictions(predictions)

            pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': high_data},
                num_epochs=1,
                shuffle=False)

            predictions = classifier.predict(input_fn=pred_input_fn)
            _, high_op = get_predictions(predictions)
        else:
            # Inclusion of code to handle nan confidence values
            low_op = classifier.predict_proba(low_data)
            nan_vals = np.isnan(low_op)
            if nan_vals.any():
                warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
                low_op[nan_vals] = 1.0/n_classes

            high_op = classifier.predict_proba(high_data)
            nan_vals = np.isnan(high_op)
            if nan_vals.any():
                warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
                high_op[nan_vals] = 1.0/n_classes

        low_op = low_op.astype('float32')
        high_op = high_op.astype('float32')

        low_op = log_loss(true_y, low_op)
        high_op = log_loss(true_y, high_op)

        pred_membership = np.where(stats.norm(0, train_loss).pdf(low_op) >= stats.norm(0, train_loss).pdf(high_op), 0, 1)
        fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
        print('AI perf for {}'.format(feature))
        print(fpr, tpr, tpr-fpr)
        attr_adv.append(tpr[1]-fpr[1])

        attr_mem.append(membership)
        attr_pred.append(np.vstack((low_op, high_op)))

    return attr_adv, attr_mem, attr_pred


def getAttributeVariations(data, feature):
    low_data, high_data = np.copy(data), np.copy(data)
    pivot = np.quantile(data[:, feature], 0.5)
    low = np.quantile(data[:, feature], 0.25)
    high = np.quantile(data[:, feature], 0.75)
    membership = np.where(data[:, feature] <= pivot, 0, 1)
    low_data[:, feature] = low
    high_data[:, feature] = high
    return low_data, high_data, membership


def better_attribute_inference(true_x, true_y, membership, classifier, train_loss, features, n_possible_vals, n_classes=None, classic_model=False):

    def getBetterAttributeVariations(data, feature, n_possible_vals):
        data_holder = []

        # Check how many unique values exist, and adjust n_possible_vals
        unique = np.unique(data[:, feature].A1)
        n_unique = len(unique)
        if n_unique < n_possible_vals:
            n_possible_vals = n_unique

        # # Compute bin edges
        # step = 1.0/n_possible_vals
        # val_bins = np.arange(0, 1, step)

        # # For all bins generate a representative value.
        # for n in range(n_possible_vals):
        #     n_data = np.copy(data)
        #     bin_mid_val = step * (n + 0.5)
        #     approx_val = np.quantile(data[:,feature], bin_mid_val)
        #     n_data[:,feature] = approx_val
        #     data_holder.append(n_data)
        # real_val_bins = np.quantile(data[:,feature], val_bins)

        # Compute bin edges
        max_val = max(unique)
        min_val = min(unique)
        step = (max_val-min_val)/n_possible_vals
        # We subtract a small amount for values landing on the bin edges
        real_val_bins = np.arange(min_val, max_val, step) - (step*0.0001)

        # For all bins generate a representative value.
        for n in range(n_possible_vals):
            n_data = np.copy(data)
            bin_mid_val = step * (n + 0.5)
            approx_val = bin_mid_val
            n_data[:,feature] = approx_val
            data_holder.append(n_data)

        # subtract 1 (-1) as first bin is 1 not 0.
        true_vec_bins = np.digitize(data[:, feature].A1, real_val_bins) - 1

        assert min(true_vec_bins) >= 0
        assert max(true_vec_bins) < n_possible_vals

        return data_holder, true_vec_bins

    print('-' * 10 + 'ATTRIBUTE INFERENCE' + '-' * 10 + '\n')

    if classic_model:
        assert n_classes is not None, "Please specify number of classes for classic models."

    attr_adv, attr_mem, attr_pred = [], [], []
    for feature in features:
        n_data, true_vec_bins = getBetterAttributeVariations(true_x, feature, n_possible_vals)

        print('Variations Gened {}'.format(feature))

        if not classic_model:
            poss_predict_holder = []
            for one_poss_data in n_data:
                pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={'x': one_poss_data},
                    num_epochs=1,
                    shuffle=False)

                predictions = classifier.predict(input_fn=pred_input_fn)
                _, poss_op = get_predictions(predictions)
                poss_op = poss_op.astype('float32')
                poss_predict_holder.append(poss_op)
        else:
            poss_predict_holder = []
            for one_poss_data in n_data:
                poss_op = classifier.predict_proba(one_poss_data)
                nan_vals = np.isnan(poss_op)

                if nan_vals.any():
                    warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
                    poss_op[nan_vals] = 1.0/n_classes

                poss_op = poss_op.astype('float32')
                poss_predict_holder.append(poss_op)

        # Compute the log loss compared to the original y-labels
        _poss_op_loss = [log_loss(true_y, poss_op) for poss_op in poss_predict_holder]
        # compute the pdf of the loss compared to train loss
        norm_obj = stats.norm(0, train_loss)
        poss_op_loss = norm_obj.pdf(_poss_op_loss)
        # Find the index with the highest loss prob (i.e. closest to train_loss)
        pred_vec_bins = np.argmax(poss_op_loss, axis=0)

        print(true_vec_bins)
        print(pred_vec_bins)

        # Ensure we are comparing the right shapes
        assert true_vec_bins.shape == pred_vec_bins.shape

        correct_prediction = (true_vec_bins == pred_vec_bins)
        value_priori = {i: 1.0*n/len(true_vec_bins) for i, n in zip(*np.unique(true_vec_bins, return_counts=True))}
        print(value_priori)
        sample_weights = np.array([value_priori[i] for i in true_vec_bins])

        # Member accuracy
        y = np.ones(len(np.where(membership == 1)[0]))
        y_dash = correct_prediction[np.where(membership == 1)[0]]
        y_weights = sample_weights[np.where(membership == 1)[0]]
        m_acc = accuracy_score(y, y_dash, sample_weight=y_weights)
        # Non-member accuracy
        y = np.ones(len(np.where(membership == 0)[0]))
        y_dash = correct_prediction[np.where(membership == 0)[0]]
        y_weights = sample_weights[np.where(membership == 0)[0]]
        nm_acc = accuracy_score(y, y_dash, sample_weight=y_weights)

        attr_adv.append(m_acc - nm_acc)
        attr_mem.append((m_acc, nm_acc))
        print(m_acc, nm_acc)

    return attr_adv, attr_mem, attr_pred


def salem_attribute_inference(true_x, true_y, membership, classifier, features, n_possible_vals, n_classes=None, classic_model=False):

    def getBetterAttributeVariations(data, feature, n_possible_vals):
        data_holder = []

        # Check how many unique values exist, and adjust n_possible_vals
        unique = np.unique(data[:, feature].A1)
        n_unique = len(unique)
        if n_unique < n_possible_vals:
            n_possible_vals = n_unique

        # Compute bin edges
        max_val = max(unique)
        min_val = min(unique)
        step = (max_val-min_val)/n_possible_vals
        # We subtract a small amount for values landing on the bin edges
        real_val_bins = np.arange(min_val, max_val, step) - (step*0.0001)

        # For all bins generate a representative value.
        for n in range(n_possible_vals):
            n_data = np.copy(data)
            bin_mid_val = step * (n + 0.5)
            approx_val = bin_mid_val
            n_data[:, feature] = approx_val
            data_holder.append(n_data)

        # subtract 1 (-1) as first bin is 1 not 0.
        true_vec_bins = np.digitize(data[:, feature].A1, real_val_bins) - 1

        assert min(true_vec_bins) >= 0
        assert max(true_vec_bins) < n_possible_vals
        assert len(data_holder[0]) == len(true_vec_bins)

        return data_holder, true_vec_bins

    print('-' * 10 + 'SALEM ATTRIBUTE INFERENCE' + '-' * 10 + '\n')

    if classic_model:
        assert n_classes is not None, "Please specify number of classes for classic models."

    attr_adv, attr_mem, attr_pred = [], [], []
    for feature in features:
        n_data, true_vec_bins = getBetterAttributeVariations(true_x, feature, n_possible_vals)

        print('Variations Gened {}'.format(feature))

        if not classic_model:
            poss_predict_holder = []
            for one_poss_data in n_data:
                pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={'x': one_poss_data},
                    num_epochs=1,
                    shuffle=False)

                predictions = classifier.predict(input_fn=pred_input_fn)
                _, poss_op = get_predictions(predictions)
                poss_op = poss_op.astype('float32')
                poss_predict_holder.append(poss_op)
        else:
            poss_predict_holder = []
            for one_poss_data in n_data:
                poss_op = classifier.predict_proba(one_poss_data)
                nan_vals = np.isnan(poss_op)

                if nan_vals.any():
                    warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
                    poss_op[nan_vals] = 1.0/n_classes

                poss_op = poss_op.astype('float32')
                poss_predict_holder.append(poss_op)

        # print(np.array(poss_predict_holder).shape)
        # print(np.argmax(np.array(poss_predict_holder), axis=0).shape)
        # print(np.argmax(np.array(poss_predict_holder), axis=1).shape)
        # print(np.max(np.array(poss_predict_holder), axis=2).shape)
        # print(np.argmax(np.max(np.array(poss_predict_holder), axis=2), axis=0).shape)
        # grab the vector with the largest probability.
        pred_vec_bins = np.argmax(np.max(np.array(poss_predict_holder), axis=2), axis=0)

        # Ensure we are comparing the right shapes
        assert true_vec_bins.shape == pred_vec_bins.shape

        correct_prediction = (true_vec_bins == pred_vec_bins)
        value_priori = {i: 1.0*n/len(true_vec_bins) for i, n in zip(*np.unique(true_vec_bins, return_counts=True))}
        print(value_priori)
        sample_weights = np.array([value_priori[i] for i in true_vec_bins])

        # Member accuracy
        y = np.ones(len(np.where(membership == 1)[0]))
        y_dash = correct_prediction[np.where(membership == 1)[0]]
        y_weights = sample_weights[np.where(membership == 1)[0]]
        m_acc = accuracy_score(y, y_dash, sample_weight=y_weights)
        # Non-member accuracy
        y = np.ones(len(np.where(membership == 0)[0]))
        y_dash = correct_prediction[np.where(membership == 0)[0]]
        y_weights = sample_weights[np.where(membership == 0)[0]]
        nm_acc = accuracy_score(y, y_dash, sample_weight=y_weights)

        attr_adv.append(m_acc - nm_acc)
        attr_mem.append((m_acc, nm_acc))
        print(m_acc, nm_acc)
    return attr_adv, attr_mem, attr_pred


def log_loss(a, b):
    return [-np.log(max(b[i, a[i]], SMALL_VALUE)) for i in range(len(a))]


def get_random_features(data, pool, size):
    # some change caused np.unique to behave oddly
    # np.unique was not returning a flattened array
    # causing an infinite loop
    features = set()
    while(len(features) < size):
        feat = random.choice(pool)
        a = np.unique(data[:, feat].A1)
        if len(a) > 1:
            features.add(feat)
    return list(features)


def run_experiment(args):
    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')

    my_res = resource_tracking(args)
    my_res.create_checkpoint('start')

    if args.alt_dataloc == 'old':
        dataset = load_data('target_data.npz')
    elif args.alt_dataloc == 'fixed':
        dataset = load_data('{}/target_data.npz'.format(args.train_dataset))
    elif args.alt_dataloc == 'sample_anew':
        dataset = sample_data_anew(args)
    else:
        assert False, 'Please specfiy where the data should be loaded from.'

    train_x, train_y, test_x, test_y, n_classes = dataset
    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)
    batch_size = args.target_batch_size

    my_res.create_checkpoint('loaded_dataset')

    # Use Tensorflow APIs
    if args.target_model in ['nn', 'regression']:
        classic_model = False
        pred_y, membership, test_classes, train_loss, classifier, train_acc, test_acc = train_target_model(
            dataset=dataset,
            epochs=args.target_epochs,
            batch_size=args.target_batch_size,
            learning_rate=args.target_learning_rate,
            n_hidden=args.target_n_hidden,
            l2_ratio=args.target_l2_ratio,
            model=args.target_model,
            privacy=args.target_privacy,
            preprocess=args.preprocess,
            dp=args.target_dp,
            epsilon=args.target_epsilon,
            delta=args.target_delta,
            save=args.save_model,
            resource_tracker=my_res)
    # Classical models following sk-learn apis
    else:
        print('USING ALTERNATE MODELS')
        print('USING ALTERNATE MODELS')
        print('USING ALTERNATE MODELS')
        classic_model = True
        pred_y, membership, test_classes, train_loss, classifier, train_acc, test_acc = train_classic_target_model(
            dataset=dataset,
            l2_ratio=args.target_l2_ratio,
            model=args.target_model,
            privacy=args.target_privacy,
            preprocess=args.preprocess,
            dp=args.target_dp,
            epsilon=args.target_epsilon,
            delta=args.target_delta,
            save=args.save_model,
            resource_tracker=my_res)

    my_res.create_checkpoint('fin_train_model')
    print('fin_train_model')

    # Choose between mlleaks MI or shadow models (very slow)
    if args.alt_mi:
        # ML-Leaks MI
        mlleak_mi_auc = mlleak_membership_inference(pred_y, membership)
        my_res.create_checkpoint('fin_mlleak-mi')
    else:
        # Shokri Shadow-Models MI
        attack_adv, attack_pred = attack_experiment(args, pred_y, membership, test_classes)
        my_res.create_checkpoint('fin_shokri-mi')

    # Somesh MI
    mem_adv, mem_pred = membership_inference(true_y, pred_y, membership, train_loss)
    my_res.create_checkpoint('fin_yeom-mi')

    # feature to be tested in AI
    features = get_random_features(true_x, range(true_x.shape[1]), 20)
    print(features)

    # Somesh AI
    if not args.better_ai:
        attr_adv, attr_mem, attr_pred = attribute_inference(true_x=true_x, true_y=true_y, 
                                                            batch_size=batch_size,
                                                            classifier=classifier,
                                                            train_loss=train_loss,
                                                            features=features,
                                                            n_classes=n_classes,
                                                            classic_model=classic_model)
        my_res.create_checkpoint('fin_yeom-ai')
    else:
        attr_adv, attr_mem, attr_pred = better_attribute_inference(true_x=true_x, true_y=true_y, 
                                                                   membership=membership,
                                                                   classifier=classifier,
                                                                   train_loss=train_loss,
                                                                   features=features,
                                                                   n_classes=n_classes,
                                                                   classic_model=classic_model,
                                                                   n_possible_vals=args.better_ai_n)
        my_res.create_checkpoint('fin_better-yeom-ai')
        print(np.mean(attr_adv))

    # Zhao AI
    zhao_ai_stats = salem_attribute_inference(true_x=true_x, true_y=true_y,
                                              membership=membership,
                                              classifier=classifier,
                                              features=features,
                                              n_classes=n_classes,
                                              classic_model=classic_model,
                                              n_possible_vals=args.better_ai_n)
    my_res.create_checkpoint('fin_better-zhao-ai')
    print(np.mean(attr_adv))

    if not os.path.exists(RESULT_PATH+args.train_dataset):
        os.makedirs(RESULT_PATH+args.train_dataset)

    if args.alt_mi:

        # Save path overhaul
        if args.target_privacy == 'noisy_data':
            target_privacy = args.target_privacy + '_' + args.preprocess
        else:
            target_privacy = args.target_privacy

        if args.target_privacy == 'no_privacy':
            level = 'level0'
            model = args.target_model
        elif args.target_privacy == 'noisy_data':
            level = 'level1'
            model = args.target_model
        elif args.target_privacy == 'grad_pert':
            level = 'level2'
            model = args.target_model
        elif args.target_model == 'dp-rndf':
            level = 'level2'
            model = 'rndf'
        elif args.target_model == 'ibmNB':
            level = 'level3'
            model = args.target_model
        elif args.target_model == 'ibmLR':
            level = 'level3'
            model = args.target_model

        save_dir = "{}{}/{}/{}/{}".format(RESULT_PATH, args.train_dataset, level, model, str(args.target_epsilon))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = "{}/{}.p".format(save_dir, args.run)

        my_res.save_log(save_dir)

        pickle.dump([train_acc,
                     test_acc,
                     train_loss,
                     membership,
                     mlleak_mi_auc,
                     zhao_ai_stats,
                     mem_adv,
                     mem_pred,
                     attr_adv,
                     attr_mem,
                     attr_pred,
                     features,
                     pred_y,
                     true_y],
                    open(save_file, 'wb'))
    else:
        assert False
        pickle.dump([train_acc,
                     test_acc,
                     train_loss,
                     membership,
                     attack_adv,
                     attack_pred,
                     mem_adv,
                     mem_pred,
                     attr_adv,
                     attr_mem,
                     attr_pred,
                     features],
                    open(RESULT_PATH + args.train_dataset + '/' + args.target_model + '_' + args.target_privacy + '_' + args.target_dp + '_' + str(args.target_epsilon) + '_' + str(args.run) + '.p', 'wb'))


# Alternate pure python entry point
def main_experiment(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    parser.add_argument('--perturb_data', type=bool, default=0)
    parser.add_argument('--alt_dataloc', type=str, default='sample_anew')
    parser.add_argument('--preprocess', type=str, default=None)
    parser.add_argument('--ignore_shadow', type=bool, default=True)
    parser.add_argument('--better_ai', type=bool, default=True)
    parser.add_argument('--better_ai_n', type=int, default=10)
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.2)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-4)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='dp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-100)
    # attack model configuration
    parser.add_argument('--alt_mi', type=bool, default=True)
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args(argv.split(' '))
    print(vars(args))
    if args.save_data:
        save_data(args)
    else:
        run_experiment(args)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    parser.add_argument('--perturb_data', type=bool, default=0)
    parser.add_argument('--alt_dataloc', type=str, default='sample_anew')
    parser.add_argument('--preprocess', type=str, default=None)
    parser.add_argument('--ignore_shadow', type=bool, default=True)
    parser.add_argument('--better_ai', type=bool, default=True)
    parser.add_argument('--better_ai_n', type=int, default=10)
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.2)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-4)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='dp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-100)
    # attack model configuration
    parser.add_argument('--alt_mi', type=bool, default=True)
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    if args.save_data:
        save_data(args)
    else:
        run_experiment(args)
