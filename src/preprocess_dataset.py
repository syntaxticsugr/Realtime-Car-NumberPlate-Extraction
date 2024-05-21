import os
import time
import pandas as pd
from skimage import io
from pathlib import Path
import xml.etree.ElementTree as ET
from pascal import annotation_from_xml
from utils.create_directory import create_directory



def preprocess_dataset(dataset_dir, output_dir):

    print("\n\n")

    df_rows = []

    for (dirpath, _, filenames) in os.walk(dataset_dir, topdown=True):

        for filename in filenames:

            df_rows.append({'dir':dirpath, 'file':filename})

    df = pd.DataFrame(data=df_rows)

    images_dir = f'{output_dir}/images'
    annotations_dir = f'{output_dir}/labels'

    create_directory(images_dir)
    create_directory(annotations_dir)

    label_map = {"numberPlate": 0}

    error_files = []

    total_time = 0
    items_remaining = len(df.index)
    print(f"\033[A\033[AProcessed: 0        Remaining: {items_remaining}        Ellapsed Time: 0        Estimated Remaining Time: --:--:--        \n")

    for index in range(0, len(df.index)):

        start_time = time.time()

        filename = df['file'][index]

        if (not filename.endswith('.xml')):

            try:
                filename_without_extension = Path(filename).stem

                image_file = filename
                image_file_path = df['dir'][df['file'] == image_file].values[0]

                annotation_file = f'{filename_without_extension}.xml'
                annotation_file_path = df['dir'][df['file'] == annotation_file].values[0]

                image = io.imread(f"{image_file_path}/{image_file}")

                with open(f'{annotation_file_path}/{annotation_file}', 'r') as f:
                    annotation_voc = f.read()

                with open(f'{annotation_file_path}/{annotation_file}', 'w') as f:

                    annotation_voc = ET.fromstring(annotation_voc)

                    for obj in annotation_voc.findall('object'):
                        name = obj.find('name')
                        name.text = 'numberPlate'

                    annotation_voc = ET.tostring(annotation_voc, encoding='unicode')

                    f.write(annotation_voc)

                annotation_voc = annotation_from_xml(f'{annotation_file_path}/{annotation_file}')
                annotation_yolo = annotation_voc.to_yolo(label_map)

                io.imsave(f'{images_dir}/{image_file}', image, check_contrast=False)
                with open(f'{annotations_dir}/{filename_without_extension}.txt', 'w') as f:
                    f.write(annotation_yolo)

            except:
                error_files.append(filename)

        total_time += (time.time() - start_time)
        items_remaining -= 1
        print(f"\033[A\033[AProcessed: {index+1}        Remaining: {items_remaining}        Ellapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}        Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime(items_remaining*(total_time/((index+1)))))}        \n")

    print(error_files)



if __name__ == '__main__':

    # Required Paths
    dataset_dir = r'dataset/kaggle/uncompressed'
    output_dir = r'dataset/processed-dataset'

    preprocess_dataset(dataset_dir, output_dir)
