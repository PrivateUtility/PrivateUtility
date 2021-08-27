import sys  


from sklearn.metrics import classification_report, accuracy_score, log_loss
from collections import OrderedDict

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer

try:
    sys.path.append("./../Smooth_Random_Trees")
    from Smooth_Random_Trees import DP_Random_Forest as DP_RNDF
except ModuleNotFoundError:
    pass
    
#from privacy.analysis.rdp_accountant import compute_rdp
#from privacy.analysis.rdp_accountant import get_privacy_spent
#from privacy.optimizers import dp_optimizer


import tensorflow as tf
import numpy as np
import os
import argparse
import warnings

import diffprivlib.models as ibmDP
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as baseLR
import sklearn.linear_model as sk_lin

LOGGING = False # enables tf.train.ProfilerHook (see use below)
LOG_DIR = 'log'

# Compatibility with tf 1 and 2 APIs
try:
  AdamOptimizer = tf.train.AdamOptimizer
except:  # pylint: disable=bare-except
  AdamOptimizer = tf.optimizers.Adam  # pylint: disable=invalid-name

# optimal sigma values for RDP mechanism for the default batch size, training set size, delta and sampling ratio.
noise_multiplier = {0.01:525, 0.05:150, 0.1:70, 0.5:13.8, 1:7, 5:1.669, 10:1.056, 50:0.551, 100:0.445, 500:0.275, 1000:0.219}


def get_predictions(predictions):
    pred_y, pred_scores = [], []
    val = next(predictions, None)
    while val is not None:
        pred_y.append(val['classes'])
        pred_scores.append(val['probabilities'])
        val = next(predictions, None)
    return np.array(pred_y), np.matrix(pred_scores)


def get_model(features, labels, mode, params):
    n, n_in, n_hidden, n_out, non_linearity, model, privacy, dp, epsilon, delta, batch_size, learning_rate, l2_ratio, epochs = params
    if model == 'nn':
        #print('Using neural network...')
        input_layer = tf.reshape(features['x'], [-1, n_in])
        y = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(input_layer)
        y = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(y)
        logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(y)
    else:
        #print('Using softmax regression...')
        input_layer = tf.reshape(features['x'], [-1, n_in])
        logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(input_layer)
    
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      #"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    vector_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    scalar_loss = tf.reduce_mean(vector_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        
        if privacy == 'grad_pert':
            C = 1 # Clipping Threshold
            sigma = 0.
            if dp == 'adv_cmp':
                sigma = np.sqrt(epochs * np.log(2.5 * epochs / delta)) * (np.sqrt(np.log(2 / delta) + 2 * epsilon) + np.sqrt(np.log(2 / delta))) / epsilon # Adv Comp
            elif dp == 'zcdp':
                sigma = np.sqrt(epochs / 2) * (np.sqrt(np.log(1 / delta) + epsilon) + np.sqrt(np.log(1 / delta))) / epsilon # zCDP
            elif dp == 'rdp':
                sigma = noise_multiplier[epsilon]
            elif dp == 'dp':
                sigma = epochs * np.sqrt(2 * np.log(1.25 * epochs / delta)) / epsilon # DP
            print(sigma)
    
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                            l2_norm_clip=C,
                            noise_multiplier=sigma,
                            num_microbatches=batch_size,
                            learning_rate=learning_rate,
                            ledger=None)
            opt_loss = vector_loss
        else:
            optimizer = AdamOptimizer(learning_rate=learning_rate)
            opt_loss = scalar_loss
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels,
                     predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          eval_metric_ops=eval_metric_ops)


def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn',
          l2_ratio=1e-7, silent=True, non_linearity='relu', privacy='no_privacy',
          preprocess=None, dp='dp', epsilon=0.5, delta=1e-5, resource_tracker=None):
    pre_train_x, train_y, test_x, test_y, n_classes = dataset

    if preprocess is None:
        train_x = pre_train_x
    else:
        def preprocess_data(preprocess, x, eps):
            # Transform x, or y, but ensure the shape and datastructure is maintained
            if preprocess == 'dp-noise':
                # assert False, 'Not implemented'
                S = x.max(axis=0) - x.min(axis=0)
                B = S/(eps/x.shape[1])
                noise = np.random.laplace(0, B, (x.shape))
                assert x.shape == noise.shape
                x_dash = x + noise
                x_dash = x_dash.astype(np.float32)
                assert x_dash.shape == x.shape
            else:
                assert False, 'Invalid Preprocess'
                pass
            assert(x.shape == (x_dash.shape)), 'Pre-processing has not preserved shape of datastructure'
#             assert(x.dtype == x_dash.dtype), 'Pre-processing has not preserved datatype of datastructure {} vs. {}'.format(x.dtype, x_dash.dtype)

            return x_dash
        
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('start_preprocess')
        ##
        train_x = preprocess_data(preprocess, pre_train_x, epsilon)
        ##
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('end_preprocess')
        
        
    if resource_tracker is not None:
            resource_tracker.create_checkpoint('start_training')
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    classifier = tf.estimator.Estimator(
            model_fn=get_model,
            params = [
                train_x.shape[0],
                n_in,
                n_hidden,
                n_out,
                non_linearity,
                model,
                privacy,
                dp,
                epsilon,
                delta,
                batch_size,
                learning_rate,
                l2_ratio,
                epochs
            ])    
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)
    test_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)
    train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': pre_train_x},
        y=train_y,
        num_epochs=1,
        shuffle=False)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        num_epochs=1,
        shuffle=False)

    steps_per_epoch = train_x.shape[0] // batch_size
    orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
    rdp = compute_rdp(batch_size / train_x.shape[0], noise_multiplier[epsilon], epochs * steps_per_epoch, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    print('\nFor delta= %.5f' % delta, ',the epsilon is: %.2f\n' % eps)

    if not os.path.exists(LOG_DIR):
       os.makedirs(LOG_DIR)
    for epoch in range(1, epochs + 1):
        hooks = []
        if LOGGING:
            hooks.append(tf.train.ProfilerHook(
                output_dir=LOG_DIR,
                save_steps=30))
        # This hook will save traces of what tensorflow is doing
        # during the training of each model. View the combined trace
        # by running `combine_traces.py`

        classifier.train(input_fn=train_input_fn,
                steps=steps_per_epoch,
                hooks=hooks)
    
#         if not silent:
#             eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
#             print('Train loss after %d epochs is: %.3f' % (epoch, eval_results['loss']))

    if resource_tracker is not None:
            resource_tracker.create_checkpoint('end_training')

    eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
    train_eval_results = eval_results
    train_acc = eval_results['accuracy']
    train_loss = eval_results['loss']
    if not silent:
        print('Train accuracy is: %.3f' % (train_acc))

    eval_results = classifier.evaluate(input_fn=test_eval_input_fn)
    test_eval_results = eval_results
    test_acc = eval_results['accuracy']
    if not silent:
        print('Test accuracy is: %.3f' % (test_acc))

    if resource_tracker is not None:
        resource_tracker.create_checkpoint('end_predictions')
        
#     predictions = classifier.predict(input_fn=pred_input_fn)
    
#     pred_y, pred_scores = get_predictions(predictions)


    return classifier, None, None, train_loss, train_acc, test_acc
    
    
def train_classic(dataset, model, silent=True, privacy='no_privacy', preprocess=None, dp='dp', epsilon=0.5, delta=1e-5, resource_tracker=None):

    train_x, train_y, test_x, test_y, n_classes = dataset

    n_in = train_x.shape[1]
    n_out = n_classes

    permissible_models = {
        'ibmNB': ibmDP.GaussianNB,
        'ibmLR': ibmDP.LogisticRegression,
        'rndf': RandomForestClassifier,
        'nb': GaussianNB,
        'lr': sk_lin.LogisticRegression,
        'dp-rndf': DP_RNDF
    }
    
    model_params = {
        'ibmNB': {'epsilon': epsilon, 'bounds': list(zip(np.min(train_x, axis=0).A1, np.max(train_x, axis=0).A1))},
        'ibmLR': {'epsilon': epsilon, 'solver':'lbfgs', 'multi_class':'ovr', 'penalty':'l2', 'max_iter':1000, 'n_jobs':-1},
        'rndf': {'n_estimators': 100, 'max_depth': 15},
        'nb': {'var_smoothing': 1e-9},
        'lr': {'solver':'lbfgs', 'multi_class':'ovr', 'penalty':'l2', 'max_iter':1000, 'n_jobs':-1},
        'dp-rndf': {'epsilon': epsilon,
                    'n_labels': len(set(train_y)),
                    'num_trees': 100,
                    'MULTI_THREAD': True,
                    'pool_size': 20},
    }
    
    # Check that our input data is valid
    assert (np.isfinite(train_x).all())
    assert (np.isfinite(test_x).all())
    
    print(model)
    print(privacy)
    
    if (model == 'ibmNB') and (privacy == 'no_privacy'):
        print('Substituting regular NB for no-privacy IBM-NB')
        classifier = permissible_models['nb'](**model_params['nb'])
    elif (model == 'ibmLR') and (privacy == 'no_privacy'):
        print('Substituting regular LR for no-privacy IBM-LR')
        classifier =  permissible_models['lr'](**model_params['lr'])
    elif (model == 'dp-rndf') and (privacy == 'no_privacy'):
        print('Substituting regular RNDF for no-privacy DP-RNDF')
        params = model_params['rndf'].copy()
        # Fixed max depth due to DP implementation having a max depth of 15.
        params['max_depth'] = 15
        classifier = permissible_models['rndf'](**params)
    else:
        classifier = permissible_models[model](**model_params[model])

    print("Start Training model")
        
    if preprocess is None:
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('start_training')
        ##
        classifier.fit(train_x, train_y)
        ##
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('end_training')

    else:
        def preprocess_data(preprocess, x, eps):
            # Transform x, or y, but ensure the shape and datastructure is maintained
            if preprocess == 'dp-noise':
                # assert False, 'Not implemented'
                S = x.max(axis=0) - x.min(axis=0)
                B = S/(eps/x.shape[1])
                noise = np.random.laplace(0, B, (x.shape))
                assert x.shape == noise.shape
                x_dash = x + noise
                assert x_dash.shape == x.shape
            else:
                assert False, 'Invalid Preprocess'
                pass
            assert(x.shape == (x_dash.shape)), 'Pre-processing has not preserved shape of datastructure'

            return x_dash
        
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('start_preprocess')
        ##
        train_x_dash = preprocess_data(preprocess, train_x, epsilon)
        ##
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('end_process')

        if resource_tracker is not None:
            resource_tracker.create_checkpoint('start_training')
        ##
        classifier.fit(train_x_dash, train_y)
        ##
        if resource_tracker is not None:
            resource_tracker.create_checkpoint('end_training')


    
    print("Start finding Train Accuracy")
    # predict and process the training data
    predict_x_labels = classifier.predict(train_x)
    train_acc = accuracy_score(train_y, predict_x_labels)
    predict_x_probs = classifier.predict_proba(train_x)

    print("compute train log-loss")
    try:
        train_loss = log_loss(train_y, predict_x_probs)
    except:
        warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
        wher_isnan = np.isnan(predict_x_probs)
        predict_x_probs[wher_isnan] = 1.0/n_out
        train_loss = log_loss(train_y, predict_x_probs)

    if not silent:
        print('Train accuracy is: %.3f' % (train_acc))

    print("Start finding Test Accuracy")
        
    # predict and process the testing data
    predict_x_labels = classifier.predict(test_x)
    test_acc = accuracy_score(test_y, predict_x_labels)
    if not silent:
        print('Test accuracy is: %.3f' % (test_acc))

    if resource_tracker is not None:
        resource_tracker.create_checkpoint('end_predictions')
        
#     predict_x_probs = classifier.predict_proba(test_x)

#     print("compute test log-loss")
#     try:
#         test_loss = log_loss(test_y, predict_x_probs)
#     except:
#         warnings.warn("nan confidence values encountered, values have been filled with 1.0/n_classes", RuntimeWarning)
#         wher_isnan = np.isnan(predict_x_probs)
#         print(wher_isnan)
#         predict_x_probs[wher_isnan] = 1.0/n_out
#         test_loss = log_loss(test_y, predict_x_probs)


#     # pred_scores contains the confidence values.
#     pred_scores = predict_x_probs
#     # pred_y contains the predicted label
#     pred_y = predict_x_labels

    return classifier, None, None, train_loss, train_acc, test_acc
    


def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(test_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(test_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(vars(args))
    dataset = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    train(dataset,
          model=args.model,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          n_hidden=args.n_hidden,
          epochs=args.epochs)


if __name__ == '__main__':
    main()
