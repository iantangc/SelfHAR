import os
import gc
import pickle
import argparse
import datetime
import time
import json
import distutils.util
import pprint

import numpy as np
import tensorflow as tf
import scipy.constants
import sklearn

import data_pre_processing
import self_har_models
import self_har_utilities
import self_har_trainers
import transformations

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{tang2021selfhar,
  title={SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data},
  author={Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={5},
  number={1},
  pages={1--30},
  year={2021},
  publisher={ACM New York, NY, USA}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk
"""

LOGS_SUB_DIRECTORY = 'logs'
MODELS_SUB_DIRECTORY = 'models'


def get_parser():
    def strtobool(v):
        return bool(distutils.util.strtobool(v))


    parser = argparse.ArgumentParser(
        description='SelfHAR Training')
        
    parser.add_argument('--working_directory', default='run',
                        help='directory containing datasets, trained models and training logs')
    parser.add_argument('--config', default='sample_configs/self_har.json',
                        help='')
    
    parser.add_argument('--labelled_dataset_path', default='run/processed_datasets/motionsense_processed.pkl', type=str, 
                        help='name of the labelled dataset for training and fine-tuning')
    parser.add_argument('--unlabelled_dataset_path', default='run/processed_datasets/hhar_processed.pkl', type=str, 
                        help='name of the unlabelled dataset to self-training and self-supervised training, ignored if only supervised training is performed.')
    
    parser.add_argument('--window_size', default=400, type=int,
                        help='the size of the sliding window')
    parser.add_argument('--max_unlabelled_windows', default=40000, type=int,
                        help='')

    parser.add_argument('--use_tensor_board_logging', default=True, type=strtobool,
                        help='')
    parser.add_argument('--verbose', default=1, type=int,
                        help='verbosity level')

    return parser

def prepare_dataset(dataset_path, window_size, get_train_test_users, validation_split_proportion=0.1, verbose=1):
    if verbose > 0:
        print(f"Loading dataset at {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)
        user_datasets = dataset_dict['user_split']
        label_list = dataset_dict['label_list']

    label_map = dict([(l, i) for i, l in enumerate(label_list)])
    output_shape = len(label_list)

    har_users = list(user_datasets.keys())
    train_users, test_users = get_train_test_users(har_users)
    if verbose > 0:
        print(f'Testing users: {test_users}, Training users: {train_users}')

    np_train, np_val, np_test = data_pre_processing.pre_process_dataset_composite(
        user_datasets=user_datasets, 
        label_map=label_map, 
        output_shape=output_shape, 
        train_users=train_users, 
        test_users=test_users, 
        window_size=window_size, 
        shift=window_size//2, 
        normalise_dataset=True, 
        validation_split_proportion=validation_split_proportion,
        verbose=verbose
    )

    return {
        'train': np_train,
        'val': np_val,
        'test': np_test,
        'label_map': label_map,
        'input_shape': np_train[0].shape[1:],
        'output_shape': output_shape,
    }

def generate_unlabelled_datasets_variations(unlabelled_data_x, labelled_data_x, labelled_repeat=1, verbose=1):
    if verbose > 0:
        print("Unlabeled data shape: ", unlabelled_data_x.shape)
    
    labelled_data_repeat = np.repeat(labelled_data_x, labelled_repeat, axis=0)
    np_unlabelled_combined = np.concatenate([unlabelled_data_x, labelled_data_repeat])
    if verbose > 0:
        print(f"Unlabelled Combined shape: {np_unlabelled_combined.shape}")
    gc.collect()

    return {
        'labelled_x_repeat': labelled_data_repeat,
        'unlabelled_combined': np_unlabelled_combined
    }

def load_unlabelled_dataset(prepared_datasets, unlabelled_dataset_path, window_size, labelled_repeat, max_unlabelled_windows=None, verbose=1):
    def get_empty_test_users(har_users):
        return (har_users, [])

    prepared_datasets['unlabelled'] = prepare_dataset(unlabelled_dataset_path, window_size, get_empty_test_users, validation_split_proportion=0, verbose=verbose)['train'][0]
    if max_unlabelled_windows is not None:
        prepared_datasets['unlabelled'] = prepared_datasets['unlabelled'][:max_unlabelled_windows]
    prepared_datasets = {
        **prepared_datasets,
        **generate_unlabelled_datasets_variations(
            prepared_datasets['unlabelled'], 
            prepared_datasets['labelled']['train'][0],
            labelled_repeat=labelled_repeat
    )}
    return prepared_datasets

def get_config_default_value_if_none(experiment_config, entry, set_value=True):
    if entry in experiment_config:
        return experiment_config[entry]
    
    if entry == 'type':
        default_value = 'none'
    elif entry == 'tag':
        default_value = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    elif entry == 'previous_config_offset':
        default_value = 0
    elif entry == 'initial_learning_rate':
        default_value = 0.0003
    elif entry == 'epochs':
        default_value = 30
    elif entry == 'batch_size':
        default_value = 300
    elif entry == 'optimizer':
        default_value = 'adam'
    elif entry == 'self_training_samples_per_class':
        default_value = 10000
    elif entry == 'self_training_minimum_confidence':
        default_value = 0.0
    elif entry == 'self_training_plurality_only':
        default_value = True
    elif entry == 'trained_model_path':
        default_value = ''
    elif entry == 'trained_model_type':
        default_value = 'unknown'
    elif entry == 'eval_results':
        default_value = {}
    elif entry == 'eval_har':
        default_value = False

    if set_value:
        experiment_config[entry] = default_value
        print(f"INFO: configuration {entry} set to default value: {default_value}.")
    
    return default_value


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    current_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    working_directory = args.working_directory
    verbose = args.verbose
    use_tensor_board_logging = args.use_tensor_board_logging
    window_size = args.window_size

    if use_tensor_board_logging:
        logs_directory = os.path.join(working_directory, LOGS_SUB_DIRECTORY)
        if not os.path.exists(logs_directory):
            os.mkdir(logs_directory)
    models_directory = os.path.join(working_directory, MODELS_SUB_DIRECTORY)
    if not os.path.exists(models_directory):
        os.mkdir(models_directory)
    transform_funcs_vectorized = [
        transformations.noise_transform_vectorized, 
        transformations.scaling_transform_vectorized, 
        transformations.rotation_transform_vectorized, 
        transformations.negate_transform_vectorized, 
        transformations.time_flip_transform_vectorized, 
        transformations.time_segment_permutation_transform_improved, 
        transformations.time_warp_transform_low_cost, 
        transformations.channel_shuffle_transform_vectorized
    ]
    transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']

    prepared_datasets = {}
    labelled_repeat = 1             # TODO: improve flexibility transformation_multiple
    

    def get_fixed_split_users(har_users):   # TODO: improve flexibility
        test_users = har_users[0::5]
        train_users = [u for u in har_users if u not in test_users]
        return (train_users, test_users)


    prepared_datasets['labelled'] = prepare_dataset(args.labelled_dataset_path, window_size, get_fixed_split_users, validation_split_proportion=0.1, verbose=verbose)
    input_shape = prepared_datasets['labelled']['input_shape'] #  (window_size, 3)
    output_shape = prepared_datasets['labelled']['output_shape']


    with open(args.config, 'r') as f:
        config_file = json.load(f)
        file_tag = config_file['tag']
        experiment_configs = config_file['experiment_configs']

    if verbose > 0:
        print("Experiment Settings:")
        for i, config in enumerate(experiment_configs):
            print(f"Experiment {i}:")
            print(config)
            print("------------")
        time.sleep(5)



    for i, experiment_config in enumerate(experiment_configs):
        if verbose > 0:
            print("---------------------")
            print(f"Starting Experiment {i}: {experiment_config}")
            print("---------------------")
            time.sleep(5)
        gc.collect()
        tf.keras.backend.clear_session()
        

        
        experiment_type = get_config_default_value_if_none(experiment_config, 'type')
        if experiment_type == 'none':
            continue
            
        if get_config_default_value_if_none(experiment_config, 'previous_config_offset') == 0:
            previous_config = None
        else:
            previous_config = experiment_configs[i - experiment_config['previous_config_offset']]
            # if verbose > 0:
            #     print("Previous config", previous_config)

        tag = f"{current_time_string}_{file_tag}_{get_config_default_value_if_none(experiment_config, 'tag')}"

        if experiment_type == 'eval_har':

            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path', set_value=False) == '':
                print("ERROR Evaluation model does not exist")
                continue
                
            if get_config_default_value_if_none(previous_config, 'trained_model_type') == 'har_model':
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path'])
                model = previous_model
            elif get_config_default_value_if_none(previous_config, 'trained_model_type') == 'transform_with_har_model':
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path'])
                model = self_har_models.extract_har_model(previous_model, optimizer=optimizer, model_name=tag)

            pred = model.predict(prepared_datasets['labelled']['test'][0])
            eval_results = self_har_utilities.evaluate_model_simple(pred, prepared_datasets['labelled']['test'][1])
            if verbose > 0:
                print(eval_results)
            experiment_config['eval_results'] = eval_results

            continue

        
        initial_learning_rate = get_config_default_value_if_none(experiment_config, 'initial_learning_rate')
        epochs = get_config_default_value_if_none(experiment_config, 'epochs')
        batch_size = get_config_default_value_if_none(experiment_config, 'batch_size')
        optimizer_type = get_config_default_value_if_none(experiment_config, 'optimizer')
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)


        if experiment_type == 'transform_train':
            if 'unlabelled' not in prepared_datasets:
                prepared_datasets = load_unlabelled_dataset(prepared_datasets, args.unlabelled_dataset_path, window_size, labelled_repeat, max_unlabelled_windows=args.max_unlabelled_windows, verbose=verbose)

            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path', set_value=False) == '':
                if verbose > 0:
                    print("Creating new model...")
                core_model = self_har_models.create_1d_conv_core_model(input_shape)
            else:
                if verbose > 0:
                    print(f"Loading previous model {previous_config['trained_model_path']}")
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path'])
                core_model = self_har_models.extract_core_model(previous_model)
            
            transform_model = self_har_models.attach_multitask_transform_head(core_model, output_tasks=transform_funcs_names, optimizer=optimizer)
            transform_model.summary()
            if verbose > 0:
                print(f"Dataset for transformation discrimination shape: {prepared_datasets['unlabelled_combined'].shape}")

            multitask_transform_dataset = self_har_utilities.create_individual_transform_dataset(prepared_datasets['unlabelled_combined'], transform_funcs_vectorized)

            multitask_transform_train = (multitask_transform_dataset[0], self_har_utilities.map_multitask_y(multitask_transform_dataset[1], transform_funcs_names))
            multitask_split = self_har_utilities.multitask_train_test_split(multitask_transform_train, test_size=0.10, random_seed=42)
            multitask_train = (multitask_split[0], multitask_split[1])
            multitask_val = (multitask_split[2], multitask_split[3])


            def training_rate_schedule(epoch):
                rate = initial_learning_rate * (0.5 ** (epoch // 15))
                if verbose > 0:
                    print(f"RATE: {rate}")
                return rate

            training_schedule_callback = tf.keras.callbacks.LearningRateScheduler(training_rate_schedule)
            
            best_transform_model_file_name, last_transform_pre_train_model_file_name = self_har_trainers.composite_train_model(
                full_model=transform_model, 
                training_set=multitask_train, 
                validation_set=multitask_val, 
                working_directory=working_directory, 
                callbacks=[training_schedule_callback], 
                epochs=epochs, 
                batch_size=batch_size, 
                tag=tag, 
                use_tensor_board_logging=use_tensor_board_logging, 
                verbose=verbose
            )
            
            experiment_config['trained_model_path'] = best_transform_model_file_name
            experiment_config['trained_model_type'] = 'transform_model'

        if experiment_type == 'har_full_train' or experiment_type == 'har_full_fine_tune' or experiment_type == 'har_linear_train':

            is_core_model = False
            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path', set_value=False) == '':
                if verbose > 0:
                    print("Creating new model...")
                core_model = self_har_models.create_1d_conv_core_model(input_shape)
                is_core_model = True
            else:
                if verbose > 0:
                    print(f"Loading previous model {previous_config['trained_model_path']}")
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path'])

                if experiment_type == 'har_linear_train':
                    core_model = self_har_models.extract_core_model(previous_model)
                    is_core_model = True
                elif get_config_default_value_if_none(previous_config, 'trained_model_type') == 'har_model':
                    har_model = previous_model
                    is_core_model = False
                elif previous_config['trained_model_type'] == 'transform_with_har_model':
                    har_model = self_har_models.extract_har_model(previous_model, optimizer=optimizer, model_name=tag)
                    is_core_model = False
                else:
                    core_model = self_har_models.extract_core_model(previous_model)
                    is_core_model = True

            if is_core_model:
                if experiment_type == 'har_linear_train':
                    self_har_models.set_freeze_layers(core_model, num_freeze_layer_index=None)
                    har_model = self_har_models.attach_linear_classification_head(core_model, output_shape, optimizer=optimizer, model_name="Linear")

                elif experiment_type == 'har_full_train':
                    self_har_models.set_freeze_layers(core_model, num_freeze_layer_index=0)
                    har_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, optimizer=optimizer, num_units=1024, model_name="HAR")
                elif experiment_type == 'har_full_fine_tune':
                    self_har_models.set_freeze_layers(core_model, num_freeze_layer_index=5)
                    har_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, optimizer=optimizer, num_units=1024, model_name="HAR")
            else:
                if experiment_type == 'har_full_train':
                    self_har_models.set_freeze_layers(self_har_models.extract_core_model(har_model), num_freeze_layer_index=0)
                elif experiment_type == 'har_full_fine_tune':
                    self_har_models.set_freeze_layers(self_har_models.extract_core_model(har_model), num_freeze_layer_index=5)
            
            def training_rate_schedule(epoch):
                rate = initial_learning_rate
                if verbose > 0:
                    print(f"RATE: {rate}")
                return rate
            training_schedule_callback = tf.keras.callbacks.LearningRateScheduler(training_rate_schedule)

            best_har_model_file_name, last_har_model_file_name = self_har_trainers.composite_train_model(
                full_model=har_model, 
                training_set=prepared_datasets['labelled']['train'],
                validation_set=prepared_datasets['labelled']['val'], 
                working_directory=working_directory, 
                callbacks=[training_schedule_callback], 
                epochs=epochs, 
                batch_size=batch_size,
                tag=tag, 
                use_tensor_board_logging=use_tensor_board_logging, 
                verbose=verbose
            )

            experiment_config['trained_model_path'] = best_har_model_file_name
            experiment_config['trained_model_type'] = 'har_model'

            
        
        if experiment_type == 'self_training' or experiment_type == 'self_har':
            if 'unlabelled' not in prepared_datasets:
                prepared_datasets = load_unlabelled_dataset(prepared_datasets, args.unlabelled_dataset_path, window_size, labelled_repeat, max_unlabelled_windows=args.max_unlabelled_windows)
            
            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path', set_value=False) == '':
                print("ERROR No previous model for self-training")
                break
            else:
                if verbose > 0:
                    print(f"Loading previous model {previous_config['trained_model_path']}")
                teacher_model = tf.keras.models.load_model(previous_config['trained_model_path'])
            if verbose > 0:
                print("Unlabelled Datasete Shape", prepared_datasets['unlabelled_combined'].shape)
            unlabelled_pred_prob = teacher_model.predict(prepared_datasets['unlabelled_combined'], batch_size=batch_size)
            np_self_labelled = self_har_utilities.pick_top_samples_per_class_np(
                prepared_datasets['unlabelled_combined'], 
                unlabelled_pred_prob, 
                num_samples_per_class=get_config_default_value_if_none(experiment_config, 'self_training_samples_per_class'), 
                minimum_threshold=get_config_default_value_if_none(experiment_config, 'self_training_minimum_confidence'), 
                plurality_only=get_config_default_value_if_none(experiment_config, 'self_training_plurality_only')
            )
            

            multitask_X, multitask_transform_y, multitask_har_y = self_har_utilities.create_individual_transform_dataset(
                np_self_labelled[0], 
                transform_funcs_vectorized, 
                other_labels=np_self_labelled[1]
            )
            

            core_model = self_har_models.create_1d_conv_core_model(input_shape)
            def training_rate_schedule(epoch):
                rate = 0.0003 * (0.5 ** (epoch // 15))
                if verbose > 0:
                    print(f"RATE: {rate}")
                return rate
            training_schedule_callback = tf.keras.callbacks.LearningRateScheduler(training_rate_schedule)

            
            if experiment_type == 'self_training':
                student_pre_train_dataset = np_self_labelled

                student_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, optimizer=optimizer, model_name="StudentPreTrain")
                student_model.summary()

                pre_train_split = sklearn.model_selection.train_test_split(student_pre_train_dataset[0], student_pre_train_dataset[1], test_size=0.10, random_state=42)
                student_pre_train_split_train = (pre_train_split[0], pre_train_split[2])
                student_pre_train_split_val = (pre_train_split[1], pre_train_split[3])

            else:

                multitask_transform_y_mapped = self_har_utilities.map_multitask_y(multitask_transform_y, transform_funcs_names)
                multitask_transform_y_mapped['har'] = multitask_har_y
                self_har_train = (multitask_X, multitask_transform_y_mapped)
                student_pre_train_dataset = self_har_train\

                student_model = self_har_models.attach_multitask_transform_head(core_model, output_tasks=transform_funcs_names, optimizer=optimizer, with_har_head=True, har_output_shape=output_shape, num_units_har=1024, model_name="StudentPreTrain")
                student_model.summary()

                pre_train_split = self_har_utilities.multitask_train_test_split(student_pre_train_dataset, test_size=0.10, random_seed=42)

                student_pre_train_split_train = (pre_train_split[0], pre_train_split[1])
                student_pre_train_split_val = (pre_train_split[2], pre_train_split[3])

            
            best_student_pre_train_file_name, last_student_pre_train_file_name = self_har_trainers.composite_train_model(
                full_model=student_model,
                training_set=student_pre_train_split_train,
                validation_set=student_pre_train_split_val, 
                working_directory=working_directory, 
                callbacks=[training_schedule_callback],
                epochs=epochs, 
                batch_size=batch_size, 
                tag=tag, 
                use_tensor_board_logging=use_tensor_board_logging, 
                verbose=verbose
            )
            

            experiment_config['trained_model_path'] = best_student_pre_train_file_name
            if experiment_type == 'self_training':
                experiment_config['trained_model_type'] = 'har_model'
            else:
                experiment_config['trained_model_type'] = 'transform_with_har_model'


        if get_config_default_value_if_none(experiment_config, 'eval_har', set_value=False):
            if get_config_default_value_if_none(experiment_config, 'trained_model_type') == 'har_model':
                best_har_model = tf.keras.models.load_model(experiment_config['trained_model_path'])
            elif get_config_default_value_if_none(experiment_config, 'trained_model_type') == 'transform_with_har_model':
                previous_model = tf.keras.models.load_model(experiment_config['trained_model_path'])
                best_har_model = self_har_models.extract_har_model(previous_model, optimizer=optimizer, model_name=tag)
            else:
                continue

            pred = best_har_model.predict(prepared_datasets['labelled']['test'][0])
            eval_results = self_har_utilities.evaluate_model_simple(pred, prepared_datasets['labelled']['test'][1])
            if verbose > 0:
                print(eval_results)
            experiment_config['eval_results'] = eval_results
    
    if verbose > 0:
        print("Finshed running all experiments.")
        print("Summary:")
        for i, config in enumerate(experiment_configs):
            print(f"Experiment {i}:")
            print(config)
            print("------------")

    result_summary_path = os.path.join(working_directory, f"{current_time_string}_{file_tag}_results_summary.txt")
    with open(result_summary_path, 'w') as f:
        structured = pprint.pformat(experiment_configs, indent=4)
        f.write(structured)
    if verbose > 0:
        print("Saved results summary to ", result_summary_path)
