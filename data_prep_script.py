# Script to load subset of training data downloaded via https://github.com/rlyss/vgg-download
# and prepare for use in MMAudio Github repository as downloaded from 
# https://github.com/hkchengrex/MMAudio/tree/main/mmaudio 

import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reference_csv', type=str, help='Path of reference CSV',
                    default='../vgg-download/data/train_vggsound_part3.csv')
parser.add_argument('-i', '--input_dir', type=str, help='Path of directory of downloaded videos',
                    default='../vgg-download/videos/train/')
parser.add_argument('-o', '--output_tsv', type=str, help='Path of new TSV file to create',
                    default='./sets/vgg-custom.tsv')

# python data_prep_script.py -r <reference TSV path> -i <directory of downloaded videos> -o <output TSV path>


args = parser.parse_args()
reference_csv = args.reference_csv
input_dir = args.input_dir
output_tsv = args.output_tsv

# Step 1: Load VGGSound CSV into dict
vgg_data = {}
with open(reference_csv, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        vid_id, _, label, _ = row
        vgg_data[vid_id] = label

# Step 2: Walk custom directory
entries = []
for fname in os.listdir(input_dir):
    if fname.endswith(".mp4"):
        vid_id = fname.replace(".mp4", "")
        if vid_id in vgg_data:
            label = vgg_data[vid_id]
            full_path = os.path.join(input_dir, fname)
            entries.append((full_path, 0, 10, label)) # Assumes all videos are 10-sec clips
        else:
            print(f"[WARNING] {vid_id} not found in reference VGGSound CSV.")

# Step 3: Write custom TSV
with open(output_tsv, "w") as f:
    writer = csv.writer(f, delimiter="\t")
    for row in entries:
        writer.writerow(row)
