import glob
import re
import os
import pandas as pd
import numpy as np

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

def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    Parameters:
        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)
    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)
        
        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how = "any", inplace = True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)
    
    return user_datasets


def process_hhar_accelerometer_files(data_folder_path):
    """
    Preprocess the accelerometer files of the HHAR dataset into the 'user-list' format
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    
    """
    # print(data_folder_path)

    har_dataset = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv')) # "<PATH_TO_HHAR_DATASET>/Phones_accelerometer.csv"
    har_dataset.dropna(how = "any", inplace = True)
    har_dataset = har_dataset[["x", "y", "z", "gt","User"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    har_users = har_dataset["user-id"].unique()

    user_datasets = {}
    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data,labels)]
    
    return user_datasets