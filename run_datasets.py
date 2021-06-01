import os
import gc
import pickle
import argparse
import datetime
import requests
import zipfile
import copy
import distutils.util

import scipy.constants
import raw_data_processing

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


"""
Making use of the following datasets:
MotionSense
    @inproceedings{Malekzadeh:2019:MSD:3302505.3310068,
        author = {Malekzadeh, Mohammad and Clegg, Richard G. and Cavallaro, Andrea and Haddadi, Hamed},
        title = {Mobile Sensor Data Anonymization},
        booktitle = {Proceedings of the International Conference on Internet of Things Design and Implementation},
        series = {IoTDI '19},
        year = {2019},
        isbn = {978-1-4503-6283-2},
        location = {Montreal, Quebec, Canada},
        pages = {49--58},
        numpages = {10},
        url = {http://doi.acm.org/10.1145/3302505.3310068},
        doi = {10.1145/3302505.3310068},
        acmid = {3310068},
        publisher = {ACM},
        address = {New York, NY, USA},
        keywords = {adversarial training, deep learning, edge computing, sensor data privacy, time series analysis},
    }

HHAR
    Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor Siiger Prentow, Mikkel Baun Kjærgaard, Anind Dey, Tobias Sonne, and Mads Møller Jensen "Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition" In Proc. 13th ACM Conference on Embedded Networked Sensor Systems (SenSys 2015), Seoul, Korea, 2015. http://dx.doi.org/10.1145/2809695.2809718
"""

DATASET_METADATA = {
    'motionsense': {
        'name': 'motionsense',
        'dataset_home_page': 'https://github.com/mmalekzadeh/motion-sense/',
        'source_url': 'https://github.com/mmalekzadeh/motion-sense/blob/master/data/B_Accelerometer_data.zip?raw=true',
        'file_name': 'B_Accelerometer_data.zip',

        'default_folder_path': 'B_Accelerometer_data',
        'save_file_name': 'motionsense_processed.pkl',
        'label_list': ['sit', 'std', 'wlk', 'ups', 'dws', 'jog'],
        'label_list_full_name': ['sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'jogging'],
        'has_null_class': False,
        'sampling_rate': 50.0,
        'unit': scipy.constants.g

    },
    'hhar': {
        'name': 'hhar',
        'dataset_home_page': 'http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition',
        'source_url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip',
        'file_name': 'Activity recognition exp.zip',

        'default_folder_path': 'Activity recognition exp',
        'save_file_name': 'hhar_processed.pkl',
        'label_list': ['sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'bike'],
        'label_list_full_name': ['sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'biking'],
        'has_null_class': False,
        'sampling_rate': 150.0,
        'unit': 1,
    }
}

ORIGINAL_DATASET_SUB_DIRECTORY = 'original_datasets'
PROCESSED_DATASET_SUB_DIRECTORY = 'processed_datasets'



def get_parser():
    parser = argparse.ArgumentParser(
        description='SelfHAR datasets download and processing script')
    parser.add_argument('--working_directory', default='run',
                        help='the output directory of the downloads and processed datasets')
    parser.add_argument('--mode', default='download_and_process', 
                        choices=['download_and_process', 'download', 'process'],
                        help='the running mode of the script.\ndownload: download the dataset(s).\nprocess: process the donwloaded dataset(s)')
    parser.add_argument('--dataset', default='all', 
                        choices=['motionsense', 'hhar', 'all'], 
                        help='name of the dataset to be downloaded/processed')
    parser.add_argument('--dataset_file_path', default=None, 
                        help='the path to the downloaded dataset for processing. Default download path is used when None.')
    return parser


def download_dataset(data_directory, dataset_metadata):
    message = f"""You are going to download the '{dataset_metadata['name']}' dataset.
    Please verify that you have visited the homepage of the dataset
    (link: {dataset_metadata['dataset_home_page']} . Note: this link is not necessarily up-to-date or accurate),
    and read any other document accompanying this dataset,
    and agree to all the terms and conditions set out by the dataset authors/data collectors.
    The author of this script is not liable for any use of this script, or any use or download of the dataset.
    You agree to be the person responsible for the download and any subsequent use of the dataset.
    Please enter 'y' to agree to the terms above, in addition to any other terms previously set out.
    """
    # answer = distutils.util.strtobool(input(message))
    answer = input(message)
    if answer == 'y':
        dataset_name = dataset_metadata['name']
        dataset_url = dataset_metadata['source_url']
        file_name = dataset_metadata['file_name']

        if not os.path.exists(os.path.join(data_directory, dataset_name)):
            os.mkdir(os.path.join(data_directory, dataset_name))

        print("Donwloading ...")
        r = requests.get(dataset_url, allow_redirects=True)
        with open(os.path.join(data_directory, dataset_name, file_name), 'wb') as f:
            f.write(r.content)
        print(f"Finshed donwloading to ({os.path.join(data_directory, dataset_name, file_name)})")
    else:
        print("You did not agree to the terms.")

def process_dataset(data_directory, processed_dataset_directory, dataset_metadata, dataset_file_path=None):
    dataset_name = dataset_metadata['name']
    file_name = dataset_metadata['file_name']
    

    print("Unzipping dataset...")
    dataset_file_path = args.dataset_file_path
    if dataset_file_path is None:
        dataset_file_path = os.path.join(data_directory, dataset_name, file_name)
    
    with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_directory, dataset_name))
    
    print("Processing dataset...")
    dataset_folder_path = os.path.join(data_directory, dataset_name, dataset_metadata['default_folder_path'])
    # print("PATH", dataset_folder_path)
    if dataset_name == 'hhar':
        user_datasets = raw_data_processing.process_hhar_accelerometer_files(dataset_folder_path)
    elif dataset_name == 'motionsense':
        user_datasets = raw_data_processing.process_motion_sense_accelerometer_files(dataset_folder_path)
    else:
        print(f"Dataset {dataset_name} is not supported.")
        return

    dataset_content = copy.copy(dataset_metadata)
    dataset_content['user_split'] = user_datasets
    with open(os.path.join(processed_dataset_directory, dataset_metadata['save_file_name']), 'wb') as f:
        pickle.dump(dataset_content, f)
    print(f"Finshed Processing, saved to {os.path.join(processed_dataset_directory, dataset_metadata['save_file_name'])}.")
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.working_directory):
        os.mkdir(args.working_directory)
    dataset_directory = os.path.join(args.working_directory, ORIGINAL_DATASET_SUB_DIRECTORY)
    if not os.path.exists(dataset_directory):
        os.mkdir(dataset_directory)

    processed_dataset_directory = os.path.join(args.working_directory, PROCESSED_DATASET_SUB_DIRECTORY)
    if not os.path.exists(processed_dataset_directory):
        os.mkdir(processed_dataset_directory)

    if args.mode == 'download_and_process' or args.mode == 'download':
        if args.dataset == 'all':
            datasets = list(DATASET_METADATA.keys())
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            print(f"-------- Downloading {dataset} --------")
            download_dataset(dataset_directory, DATASET_METADATA[dataset])
    

    if args.mode == 'download_and_process' or args.mode == 'process':
        if args.dataset == 'all':
            datasets= list(DATASET_METADATA.keys())
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            print(f"-------- Processing {dataset} --------")
            process_dataset(dataset_directory, processed_dataset_directory, DATASET_METADATA[dataset], args.dataset_file_path)
    print("Finished")
        
        

