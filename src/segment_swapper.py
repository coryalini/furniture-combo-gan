from cProfile import label
from hashlib import new
from importlib.abc import TraversableResources
import json
import pytorch3d.io as io
import os
import numpy as np
import utils
import torch
import utils
from scipy.spatial.transform import Rotation as R

# file path to the stats folder inside stats
base_filepath = "/home/dupmaka/mf-combo-gan/partnet_dataset/stats"
base_partnet_path = "/home/dupmaka/shapenet_data/data_v0"

# Returns the dictionary representation of the train and validation image paths
def get_splits(obj_name, base_filepath):

    train_path = "/train_val_test_split/" + obj_name + ".train.json"
    val_path = "/train_val_test_split/" + obj_name + ".val.json"

    train = json.load(open(base_filepath + train_path))
    val = json.load(open(base_filepath + val_path))

    return train, val


def get_hierarchy(obj_name, base_partnet_path):

    hierarchy = []
    index_full = []
    idx_filename =  "/home/dupmaka/mf-combo-gan/partnet_dataset/stats/after_merging_label_ids/" + obj_name +"-level-1.txt"

    f = open(idx_filename, "r")
    lines = f.readlines()
 
    for line in lines:
        line = line.split()
        parts = line[1].split("/")
        part = parts[1]
        hierarchy.append(part)
        index_full.append(parts[0])
    
    return hierarchy, index_full


def get_labels(base_path, id, label_map, obj_name):

    label_map = label_map(base_path, id, obj_name)

    label_path = base_path + "/" + str(id) + "/point_sample/sample-points-all-label-10000.txt"
    point_path = base_path + "/" + str(id) + "/point_sample/sample-points-all-pts-nor-rgba-10000.txt"

    pts, nor, rgb, opacity = load_file(point_path)
    pts = normalize_pc(pts)

    labels = load_label(label_path)

    return pts, labels


# from partnet_dataset/scripts/gen_h5_ins_seg_after_merging.py 
def load_file(fn):
    with open(fn, 'r') as fin:
        lines = [line.rstrip().split() for line in fin.readlines()]
        pts = np.array([[float(line[0]), float(line[1]), float(line[2])] for line in lines], dtype=np.float32)
        nor = np.array([[float(line[3]), float(line[4]), float(line[5])] for line in lines], dtype=np.float32)
        rgb = np.array([[int(line[6]), int(line[7]), int(line[8])] for line in lines], dtype=np.float32)
        opacity = np.array([float(line[9]) for line in lines], dtype=np.float32)
        return pts, nor, rgb, opacity

# from partnet_dataset/scripts/gen_h5_ins_seg_after_merging.py 
def normalize_pc(pts):
    x_max = np.max(pts[:, 0]); x_min = np.min(pts[:, 0]); x_mean = (x_max + x_min) / 2
    y_max = np.max(pts[:, 1]); y_min = np.min(pts[:, 1]); y_mean = (y_max + y_min) / 2
    z_max = np.max(pts[:, 2]); z_min = np.min(pts[:, 2]); z_mean = (z_max + z_min) / 2
    pts[:, 0] -= x_mean
    pts[:, 1] -= y_mean
    pts[:, 2] -= z_mean
    scale = np.sqrt(np.max(np.sum(pts**2, axis=1)))
    pts /= scale
    return pts

# from partnet_dataset/scripts/gen_h5_ins_seg_after_merging.py 
def load_label(fn):
    with open(fn, 'r') as fin:
        label = np.array([int(item.rstrip()) for item in fin.readlines()], dtype=np.int32)
        return label

def label_map(base_path, id, obj_name):

    label_map = {}

    hierarchy, index_full = get_hierarchy(obj_name, base_partnet_path)

    traverse_path = base_path + "/" + str(id) + "/parts_render_after_merging"

    for filename in os.listdir(traverse_path):
        f = os.path.join(traverse_path, filename)
        
        # checking if it is a file
        if f[-3:] == "txt":
            fi = open(f, "r")
            line = fi.readline()
            line = line.split()
            idx = line[0]
            part_name= line[1]

            if part_name in hierarchy:
                label_map[part_name] = idx

    return label_map

def get_sub_pc(id, base_partnet_path, obj_name):
    ids = str(obj_name) + "_" + str(id)
    new_pc = []
    label_path = base_partnet_path + "/" + str(id) + "/point_sample/sample-points-all-label-10000.txt"
    point_path = base_partnet_path + "/" + str(id) + "/point_sample/sample-points-all-pts-nor-rgba-10000.txt"

    pts, nor, rgb, opacity = load_file(point_path)
    pts = normalize_pc(pts)

    labels = load_label(label_path)

    point_label_dict = {}
    lengths = {}
    for i, l in enumerate(labels):
        try:
            point_label_dict[l].append(i)
            lengths[l] += 1
        except:
            point_label_dict[l] = []
            point_label_dict[l].append(i)
            lengths[l] = 1

    # Use this to get maximum occuring categories
    lengths_sorted = lengths# {k: v for k, v in sorted(lengths.items(), key=lambda w: w[1], reverse=True)}

    # get index values at the max labeled points
    if obj_name == "Chair":
        keys = list(lengths_sorted.keys())
    else:
        lengths_sorted = {k: v for k, v in sorted(lengths.items(), key=lambda w: w[1], reverse=True)}
        keys = list(lengths_sorted.keys())
        keys = keys[:len(keys)//3]
        
    for k in keys:
        max_index = point_label_dict[k]

        # select these points to consider
        pts_selected = pts[max_index]

        new_pc.append(pts_selected)
     
    try:
        new_pc = np.concatenate(new_pc, axis=0)
    except:
        return [], ids

    if obj_name != "Chair":
        # rotate table by 90 degrees
        # Robustness Test: Rotation, 90 z, 180 z, 90x+90y
        new_pc[:, 0] = new_pc[:, 0]+.7
        new_pc[:, 1] = new_pc[:, 1]+.1
        ranges = np.linspace(-np.pi, np.pi, 10)
        angle = np.random.choice(ranges)
        r = R.from_rotvec([0, angle, 0])
        r = r.as_matrix()
        new_pc = np.matmul(new_pc, r)

        # new_pc = normalize_pc(new_pc)

    return new_pc, ids

obj_name_1 = "Chair"
obj_name_2 = "Table"
train_1, val_1 = get_splits(obj_name_1, base_filepath)
train_2, val_2 = get_splits(obj_name_2, base_filepath)

count = 0
combined_pts = []
combined_labels = []

new_pc = []
ids = ""

train = train_1
if len(train_1) > len(train_2):
    train = train_2

for i, t in enumerate(train): 
    id_1 = train_1[i]['anno_id']
    id_2 = train_2[i]['anno_id']
    
    pc_1, name_1 = get_sub_pc(id_1, base_partnet_path, obj_name_1)
    pc_2, name_2 = get_sub_pc(id_2, base_partnet_path, obj_name_2)

    if len(pc_1) == 0 and len(pc_2) > 0:
        new_pc = torch.from_numpy(pc_2)
    elif len(pc_1) > 0 and len(pc_2) == 0:
        new_pc = torch.from_numpy(pc_1)
    elif len(pc_1) > 0 and len(pc_2) > 0:
        new_pc = torch.from_numpy(np.concatenate([pc_1, pc_2], axis=0))
    else:
        continue
    utils.viz_point_cloud(new_pc, "../data/combined_images/"+ name_1 + "_" + name_2 +".gif", "cuda", new_pc.shape[0])
    np.save("../data/combined_pc/"+ name_1 + "_" + name_2 +".npy", new_pc)

    # if count < 2:
    #     ids += str(id) + " "
    #     label_path = base_partnet_path + "/" + str(id) + "/point_sample/sample-points-all-label-10000.txt"
    #     point_path = base_partnet_path + "/" + str(id) + "/point_sample/sample-points-all-pts-nor-rgba-10000.txt"

    #     pts, nor, rgb, opacity = load_file(point_path)
    #     pts = normalize_pc(pts)

    #     labels = load_label(label_path)

    #     point_label_dict = {}
    #     lengths = {}
    #     for i, l in enumerate(labels):
    #         try:
    #             point_label_dict[l].append(i)
    #             lengths[l] += 1
    #         except:
    #             point_label_dict[l] = []
    #             point_label_dict[l].append(i)
    #             lengths[l] = 1

    #     # Use this to get maximum occuring categories
    #     lengths_sorted =  {k: v for k, v in sorted(lengths.items(), key=lambda w: w[1], reverse=True)}

    #     # get index values at the max labeled points
    #     keys = list(lengths_sorted.keys())[::2]

    #     for k in keys:
    #         max_index = point_label_dict[k]

    #         # select these points to consider
    #         pts_selected = pts[max_index]

    #         new_pc.append(pts_selected)
    #     count += 1
    # else:
    #     new_pc = torch.from_numpy(np.concatenate(new_pc, axis=0))
    #     utils.viz_point_cloud(new_pc, "images/"+ str(ids)+".gif", "cuda", new_pc.shape[0])
    #     ids = ""
    #     count = 0
    #     new_pc = []



# for t in train:
#     if count < 3:
#         id = t['anno_id']
#         print(id)
#         label = label_map(base_partnet_path, id, obj_name)
#         print(label)
#         pts, labels = get_labels(base_partnet_path, id, label_map, obj_name)
        # if count == 0:
        #     idx = label['chair_seat']
        #     print(type(idx))
        # elif count == 1:
        #     idx = label['chair_back']

        # elif count == 2:
        #     idx = label['chair_base']

        # print(idx)
        # print(labels)
        # exit()
        # labels_int = []
        # index = []
        # for i, l in enumerate(labels):
        #     labels_int.append(int(l))
        #     print(idx, l)
        #     if int(l) == int(idx):
        #         index.append(i)
        #         print(i, l)
        # labels_int = np.array(labels_int)

        # print(index)
        # # print(np.argwhere(labels_int==int(idx)))
        # combined_pts.append(pts[index])
        # # combined_labels.append()
        # count += 1

# combined_pts = np.array(combined_pts)
# print(combined_pts.shape)

