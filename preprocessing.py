import json
import shutil
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xmltodict
import dicttoxml


base_path = Path('/mnt/ssd0_1/kashin/gr_mri_spine')
descr_path = base_path / 'descr'
imgs_path = base_path / 'images'


def load_data():
    return pd.concat(pd.read_csv(path) for path in descr_path.iterdir())


def select_using_data(df):
    df = df[df['На срезе визуализируются межпозвоночные диски']
            == 'Визуализируются (можно размечать)']
    df = df[~df.XML.isnull()]
    return df


def parse_row(row):
    path = row['Файлы'][:-2]
    path = imgs_path / path
    xml = xmltodict.parse(row['XML'])
    if 'annotationgroup' in xml:
        xml = xml['annotationgroup']
    objects = xml['annotation']['object']
    return path, objects


def create_label_indx_mappings(df):
    labels = []
    for row in df.iterrows():
        row = row[1]
        path, objects = parse_row(row)
        labels.extend([o['name'] for o in objects])

    vc = pd.Series.value_counts(labels)
    using_labels = vc[vc > 1].index.tolist()  # filter rare labels
    print(vc)
    print()
    print(f'Using labels: {using_labels}')

    indx2labels = dict(enumerate(using_labels))
    labels2indx = {y: x for x, y in indx2labels.items()}
    indx2color = plt.cm.get_cmap('gist_rainbow', len(using_labels))
    return using_labels, indx2labels, labels2indx, indx2color


def draw_objects(img, objs, labels2indx, indx2color):
    img = img.copy()
    for o in objs:
        label = labels2indx[o['name']]
        c = np.array(indx2color(label)) * 255
        pt = o['polygon']['pt']
        a, b = pt[0], pt[2]
        img = cv2.rectangle(img, (int(a['x']), int(a['y'])),
                            (int(b['x']), int(b['y'])), c, 2)
    return img


def get_bounding_box(obj, labels2indx):
    if obj['name'] not in labels2indx:
        return None

    bb_obj = {'label': labels2indx[obj['name']]}
    pt = obj['polygon']['pt']
    bb_obj['xmin'] = pt[0]['x']
    bb_obj['xmax'] = pt[1]['x']
    bb_obj['ymin'] = pt[0]['y']
    bb_obj['ymax'] = pt[2]['y']
    return bb_obj


def preprocess_bounding_boxes(objs, labels2indx):
    bboxes = []
    for obj in objs:
        bb = get_bounding_box(obj, labels2indx)
        if bb is not None:
            bboxes.append(bb)
    return bboxes


def split_data(df, labels2indx, folds=7):
    all_paths = []
    all_objects = []
    split_labels = []
    for row in df.iterrows():
        row = row[1]
        path, objects = parse_row(row)
        all_paths.append(path)
        all_objects.append(preprocess_bounding_boxes(objects, labels2indx))

        # list of all labels in image
        classes = [labels2indx.get(o['name'], -1) for o in objects]

        # train-val-test split based on the most rare label on image
        # greater index mean more rare label
        split_labels.append(max(classes))

    skf = StratifiedKFold(n_splits=folds, random_state=42)
    train_inds = []
    val_inds = []
    test_inds = []

    for i, split in enumerate(skf.split(split_labels, split_labels)):
        if i == 0:
            val_inds = list(split[1])
        elif i == 1:
            test_inds = list(split[1])
        else:
            train_inds += list(split[1])

    base_path.joinpath(f'split_{folds}.json').write_text(
        json.dumps({
            'train_inds': train_inds,
            'val_inds': val_inds,
            'test_inds': test_inds
        }, indent=True, sort_keys=True, default=int)
    )

    train_paths = select_fold(all_paths, train_inds)
    val_paths = select_fold(all_paths, val_inds)
    test_paths = select_fold(all_paths, test_inds)

    train_objs = select_fold(all_objects, train_inds)
    val_objs = select_fold(all_objects, val_inds)
    test_objs = select_fold(all_objects, test_inds)

    return train_paths, val_paths, test_paths, train_objs, val_objs, test_objs


def select_fold(items, inds):
    return [item for i, item in enumerate(items) if i in inds]


def obj_to_voc(path, obj):
    item = {'filename': path.name}
    img = imageio.imread(path)
    item['width'] = img.shape[1]
    item['height'] = img.shape[0]

    item['object'] = []
    for o in obj:
        o = o.copy()
        item['object'].append({
            'name': o.pop('label'),
            'bndbox': o
        })
    return dicttoxml.dicttoxml(item, attr_type=False, custom_root='annotation',
                               fold_list=False)


def save_to_voc(paths, objs, folder):
    for p, o in zip(paths, objs):
        o_xml = obj_to_voc(p, o)
        path_xml = folder / (p.name.split('.')[0] + '.xml')
        path_xml.write_bytes(o_xml)


def copy_imgs(paths, folder):
    for path in paths:
        shutil.copy(path, folder / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_folds', default=7, type=int)
    args = parser.parse_args()

    data = load_data()
    data = select_using_data(data)

    using_labels, indx2labels, labels2indx, indx2color = create_label_indx_mappings(data)
    train_paths, val_paths, test_paths, train_objs, val_objs, test_objs = split_data(
        data, labels2indx, folds=args.k_folds)

    train_imgs_folder = base_path / 'train_imgs'
    train_anns_folder = base_path / 'train_annotations'
    train_imgs_folder.mkdir(exist_ok=True)
    train_anns_folder.mkdir(exist_ok=True)

    val_imgs_folder = base_path / 'val_imgs'
    val_anns_folder = base_path / 'val_annotations'
    val_imgs_folder.mkdir(exist_ok=True)
    val_anns_folder.mkdir(exist_ok=True)

    test_imgs_folder = base_path / 'test_imgs'
    test_anns_folder = base_path / 'test_annotations'
    test_imgs_folder.mkdir(exist_ok=True)
    test_anns_folder.mkdir(exist_ok=True)

    save_to_voc(train_paths, train_objs, train_anns_folder)
    save_to_voc(val_paths, val_objs, val_anns_folder)
    save_to_voc(test_paths, test_objs, test_anns_folder)

    copy_imgs(train_paths, train_imgs_folder)
    copy_imgs(val_paths, val_imgs_folder)
    copy_imgs(test_paths, test_imgs_folder)


if __name__ == '__main__':
    main()
