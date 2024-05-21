import os
import sys
import zipfile



def download_dataset(kaggle_username, kaggle_key, datasets, destination_path):

    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    for dataset in datasets:
        api.dataset_download_files(dataset, path=destination_path, quiet=False, unzip=False)



def unzip_dataset(datasets, destination_path):

    for dataset in datasets:

        archive = f'{destination_path}/{os.path.basename(dataset)}.zip'
        extract_to = f'{destination_path}/uncompressed/{os.path.basename(dataset)}'

        with zipfile.ZipFile(archive, 'r') as zip_ref:

            file_list = zip_ref.namelist()

            total_files = len(file_list)

            files_extracted = 0

            for file in file_list:
                zip_ref.extract(file, extract_to)
                files_extracted += 1

                progress = (files_extracted / total_files) * 100
                sys.stdout.write(f"\rProgress: {progress:.2f}% ({files_extracted}/{total_files} files extracted)")
                sys.stdout.flush()



if __name__ == "__main__":

    kaggle_username = ''
    kaggle_key = ''

    datasets = [
        'andrewmvd/car-plate-detection',
        'saisirishan/indian-vehicle-dataset',
    ]

    destination_path = r'dataset/kaggle'

    download_dataset(kaggle_username, kaggle_key, datasets, destination_path)

    unzip_dataset(datasets, destination_path)
