
import os
import shutil

from tqdm import tqdm

import argparse

def transfer_0_to_1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_files', type=int, default=5000, help='Number of files to copy')
    args = parser.parse_args()

    base = '/home/e12223411/repos/difusco/data/mis'

    train = os.path.join(base, 'er_train')
    train_0_ann = os.path.join(base, 'er_train_annotations_0')
    train_1 = os.path.join(base, 'er_train_1')
    os.makedirs(train_1, exist_ok=True)
    train_1_ann = os.path.join(base, 'er_train_annotations_1')
    os.makedirs(train_1_ann, exist_ok=True)
    
    # copy files from train to train_1 if they are not in train_0
    # take name on the left of the last _
    graphs_0 = {'_'.join(g.split('_')[:-1]) for g in os.listdir(train_0_ann) if g.endswith('unweighted.result')}
    print('graphs_0:', len(graphs_0))
    graphs_1 = [g for g in os.listdir(train) if g[:-len('.gpickle')] not in graphs_0 and g.endswith('.gpickle')]
    print('graphs_1:', len(graphs_1))

    print(len(graphs_1) + len(graphs_0))
    print(len([x for x in os.listdir(train) if x.endswith('.gpickle')]))

    # Copy num_files 1 graphs from train to train_1
    for g in tqdm(graphs_1[:args.num_files]):
        shutil.copy(os.path.join(train, g), os.path.join(train_1, g))

    print('train:', len(os.listdir(train)))
    print('train_0_ann:', len(os.listdir(train_0_ann)))
    print('train_1:', len(os.listdir(train_1)))
    print('train_1_ann:', len(os.listdir(train_1_ann)))


def rename_train_degree_labels():
    base = '/home/e12223411/repos/difusco/data/mis'
    train_ann = os.path.join(base, 'er_train_degree_labels')

    example = os.path.join(base, "er_train_annotations_1")
    files = {'_'.join(x.removesuffix('_unweighted.result').split('_')[:-1]) for x in os.listdir(example)}
    print(files)
    prefix = files.pop()

    # add the prefix to all files in train_ann

    for f in tqdm(os.listdir(train_ann)):
        if f.endswith('.txt'):
            new_f = f"{prefix}_{f.removesuffix('.txt')}_unweighted.result"
            assert 0 <= int(f.removesuffix('.txt')) <= 170000
            os.rename(os.path.join(train_ann, f), os.path.join(train_ann, new_f))

def move_annotations():
    base = '/home/e12223411/repos/difusco/data/mis'
    src_dir = os.path.join(base, 'er_train_annotations_1')
    dest_dir = os.path.join(base, 'er_train_annotations_0')

    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Move all files from src_dir to dest_dir
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        shutil.move(src_file, dest_file)


def check_for_gaps():
    base = '/home/e12223411/repos/difusco/data/mis'
    train = os.path.join(base, 'er_train')
    print(len(os.listdir(train)))
    train_ann = os.path.join(base, 'er_train_annotations_mis')
    print(len(os.listdir(train_ann)))


    # extract id:
    def extract_id(s):
        return int(s.split(".")[1].split("_")[1])
    
    # missing in train
    ids = sorted(extract_id(f) for f in os.listdir(train_ann) if f.endswith('.result'))
    missing = set(range(1, len(train_ann) + 1)) - set(ids)
    print(f"missing in train: {missing}")

    # missing in train_ann
    ids = sorted(extract_id(f) for f in os.listdir(train) if f.endswith('.gpickle'))
    missing = set(range(1, len(train) + 1)) - set(ids)
    print(f"missing in train_ann: {missing}")


if __name__ == '__main__':
    # move_annotations()
    transfer_0_to_1()
    # rename_train_degree_labels()
    # check_for_gaps()