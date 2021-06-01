import numpy as np
import sklearn
import gc

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{10.1145/3448112,
  author = {Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  title = {SelfHAR: Improving Human Activity Recognition through Self-Training with Unlabeled Data},
  year = {2021},
  issue_date = {March 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {5},
  number = {1},
  url = {https://doi.org/10.1145/3448112},
  doi = {10.1145/3448112},
  abstract = {Machine learning and deep learning have shown great promise in mobile sensing applications, including Human Activity Recognition. However, the performance of such models in real-world settings largely depends on the availability of large datasets that captures diverse behaviors. Recently, studies in computer vision and natural language processing have shown that leveraging massive amounts of unlabeled data enables performance on par with state-of-the-art supervised models.In this work, we present SelfHAR, a semi-supervised model that effectively learns to leverage unlabeled mobile sensing datasets to complement small labeled datasets. Our approach combines teacher-student self-training, which distills the knowledge of unlabeled and labeled datasets while allowing for data augmentation, and multi-task self-supervision, which learns robust signal-level representations by predicting distorted versions of the input.We evaluated SelfHAR on various HAR datasets and showed state-of-the-art performance over supervised and previous semi-supervised approaches, with up to 12% increase in F1 score using the same number of model parameters at inference. Furthermore, SelfHAR is data-efficient, reaching similar performance using up to 10 times less labeled data compared to supervised approaches. Our work not only achieves state-of-the-art performance in a diverse set of HAR datasets, but also sheds light on how pre-training tasks may affect downstream performance.},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  month = mar,
  articleno = {36},
  numpages = {30},
  keywords = {semi-supervised training, human activity recognition, unlabeled data, self-supervised training, self-training, deep learning}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk

Copyright (C) 2021 C. I. Tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

def create_individual_transform_dataset(X, transform_funcs, other_labels=None, multiple=1, is_transform_func_vectorized=True, verbose=1):
    label_depth = len(transform_funcs)
    transform_x = []
    transform_y = []
    other_y = []
    if is_transform_func_vectorized:
        for _ in range(multiple):
            
            transform_x.append(X)
            ys = np.zeros((len(X), label_depth), dtype=int)
            transform_y.append(ys)
            if other_labels is not None:
                other_y.append(other_labels)

            for i, transform_func in enumerate(transform_funcs):
                if verbose > 0:
                    print(f"Using transformation {i} {transform_func}")
                transform_x.append(transform_func(X))
                ys = np.zeros((len(X), label_depth), dtype=int)
                ys[:, i] = 1
                transform_y.append(ys)
                if other_labels is not None:
                    other_y.append(other_labels)
        if other_labels is not None:
            return np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0), np.concatenate(other_y, axis=0)
        else:
            return np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0), 
    else:
        for _ in range(multiple):
            for i, sample in enumerate(X):
                if verbose > 0 and i % 1000 == 0:
                    print(f"Processing sample {i}")
                    gc.collect()
                y = np.zeros(label_depth, dtype=int)
                transform_x.append(sample)
                transform_y.append(y)
                if other_labels is not None:
                    other_y.append(other_labels[i])
                for j, transform_func in enumerate(transform_funcs):
                    y = np.zeros(label_depth, dtype=int)
                    # transform_x.append(sample)
                    # transform_y.append(y.copy())

                    y[j] = 1
                    transform_x.append(transform_func(sample))
                    transform_y.append(y)
                    if other_labels is not None:
                        other_y.append(other_labels[i])
        if other_labels is not None:
            np.stack(transform_x), np.stack(transform_y), np.stack(other_y)
        else:
            return np.stack(transform_x), np.stack(transform_y)

def map_multitask_y(y, output_tasks):
    multitask_y = {}
    for i, task in enumerate(output_tasks):
        multitask_y[task] = y[:, i]
    return multitask_y


def multitask_train_test_split(dataset, test_size=0.1, random_seed=42):
    dataset_size = len(dataset[0])
    indices = np.arange(dataset_size)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    test_dataset_size = int(dataset_size * test_size)
    return dataset[0][indices[test_dataset_size:]], dict([(k, v[indices[test_dataset_size:]]) for k, v in dataset[1].items()]), dataset[0][indices[:test_dataset_size]], dict([(k, v[indices[:test_dataset_size]]) for k, v in dataset[1].items()])

def evaluate_model_simple(pred, truth, is_one_hot=True, return_dict=True):
    """
    Evaluate the prediction results of a model with 7 different metrics
    Metrics:
        Confusion Matrix
        F1 Macro
        F1 Micro
        F1 Weighted
        Precision
        Recall 
        Kappa (sklearn.metrics.cohen_kappa_score)
    Parameters:
        pred
            predictions made by the model
        truth
            the ground-truth labels
        
        is_one_hot=True
            whether the predictions and ground-truth labels are one-hot encoded or not
        return_dict=True
            whether to return the results in dictionary form (return a tuple if False)
    Return:
        results
            dictionary with 7 entries if return_dict=True
            tuple of size 7 if return_dict=False
    """

    if is_one_hot:
        truth_argmax = np.argmax(truth, axis=1)
        pred_argmax = np.argmax(pred, axis=1)
    else:
        truth_argmax = truth
        pred_argmax = pred

    test_cm = sklearn.metrics.confusion_matrix(truth_argmax, pred_argmax)
    test_f1 = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='macro')
    test_precision = sklearn.metrics.precision_score(truth_argmax, pred_argmax, average='macro')
    test_recall = sklearn.metrics.recall_score(truth_argmax, pred_argmax, average='macro')
    test_kappa = sklearn.metrics.cohen_kappa_score(truth_argmax, pred_argmax)

    test_f1_micro = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='micro')
    test_f1_weighted = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='weighted')

    if return_dict:
        return {
            'Confusion Matrix': test_cm, 
            'F1 Macro': test_f1, 
            'F1 Micro': test_f1_micro, 
            'F1 Weighted': test_f1_weighted, 
            'Precision': test_precision, 
            'Recall': test_recall, 
            'Kappa': test_kappa
        }
    else:
        return (test_cm, test_f1, test_f1_micro, test_f1_weighted, test_precision, test_recall, test_kappa)


def pick_top_samples_per_class_np(X, y_prob, num_samples_per_class=500, minimum_threshold=0, plurality_only=False, verbose=1):
    is_sample_selected_overall = np.full(len(X), False, dtype=bool)
    num_classes = y_prob.shape[-1]

    for c in range(num_classes):
        if verbose > 0:
            print(f"---Processing class {c}---")
        is_sample_selected_class = np.full(len(X), True, dtype=bool)

        if plurality_only:
            is_sample_selected_class = (np.argmax(y_prob, axis=1) == c) & is_sample_selected_class
            if verbose > 0:
                print(f"Passes plurality test: {np.sum(is_sample_selected_class)}")
        is_sample_selected_class = (y_prob[:, c] >= minimum_threshold) & is_sample_selected_class
        if verbose > 0:
            print(f"Passes minimum threshold: {np.sum(is_sample_selected_class)}")

        current_selection_count = np.sum(is_sample_selected_class)

        if current_selection_count == 0:
            if verbose > 0:
                print(f"No sample is above threshold {minimum_threshold}")
                continue
        if current_selection_count > num_samples_per_class:
            masked_y_prob = np.where(is_sample_selected_class, y_prob[:,c], 0)
            selection_indices = np.argpartition(-masked_y_prob, num_samples_per_class)

            is_sample_selected_class[selection_indices[:num_samples_per_class]] = True
            is_sample_selected_class[selection_indices[num_samples_per_class:]] = False
            if verbose > 0:
                print(f"Final selection for class: {np.sum(is_sample_selected_class)}, with minimum confidence : {y_prob[selection_indices[num_samples_per_class-1],c]}")
        else:
            if verbose > 0:
                print(f"Final selection for class: {np.sum(is_sample_selected_class)}")
        
        is_sample_selected_overall = is_sample_selected_class | is_sample_selected_overall
        if verbose > 0:
            print(f"Currnt total selection: {np.sum(is_sample_selected_overall)}")
    return X[is_sample_selected_overall], y_prob[is_sample_selected_overall]

