
import os
import shutil

from tqdm import tqdm


def transfer_to_31oct():
    base = '/home/e12223411/repos/difusco/data/mis'

    train = os.path.join(base, 'er_train')
    train_28oct_ann = os.path.join(base, 'er_train_annotations_28oct')
    train_ann = os.path.join(base, 'er_train_annotations_31oct')
    train_31oct = os.path.join(base, 'er_train_31oct')
    train_31oct_ann = os.path.join(base, 'er_train_annotations_31oct')

    # copy files from train to train_31oct if they are not in train_28oct
    # take name on the left of the last _
    graphs_28oct = {'_'.join(g.split('_')[:-1]) for g in os.listdir(train_28oct_ann) if g.endswith('unweighted.result')}
    print('graphs_28oct:', len(graphs_28oct))
    graphs_31oct = [g for g in os.listdir(train) if g[:-len('.gpickle')] not in graphs_28oct and g.endswith('.gpickle')]
    print('graphs_31oct:', len(graphs_31oct))

    print(len(graphs_31oct) + len(graphs_28oct))
    print(len(os.listdir(train)))

    # Copy all 31oct graphs from train to train_31oct
    for g in tqdm(graphs_31oct):
        shutil.copy(os.path.join(train, g), os.path.join(train_31oct, g))

    print('train:', len(os.listdir(train)))
    print('train_ann:', len(os.listdir(train_ann)))
    print('train_28oct_ann:', len(os.listdir(train_28oct_ann)))
    print('train_31oct:', len(os.listdir(train_31oct)))
    print('train_31oct_ann:', len(os.listdir(train_31oct_ann)))


def rename_train_degree_labels():
    base = '/home/e12223411/repos/difusco/data/mis'
    train_ann = os.path.join(base, 'er_train_degree_labels')

    example = os.path.join(base, "er_train_annotations_31oct")
    files = {'_'.join(x.removesuffix('_unweighted.result').split('_')[:-1]) for x in os.listdir(example)}
    print(files)
    prefix = files.pop()

    # add the prefix to all files in train_ann

    for f in tqdm(os.listdir(train_ann)):
        if f.endswith('.txt'):
            new_f = f"{prefix}_{f.removesuffix('.txt')}_unweighted.result"
            assert 0 <= int(f.removesuffix('.txt')) <= 170000
            os.rename(os.path.join(train_ann, f), os.path.join(train_ann, new_f))


if __name__ == '__main__':
    # transfer_to_31oct()
    rename_train_degree_labels()