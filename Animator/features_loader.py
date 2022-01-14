import pickle
import numpy as np
import pandas as pd
import os


def deserialize_feature_vector(serialized_vector):
    deserialized_vec = pickle.loads(serialized_vector)
    return deserialized_vec


def serialize_feature_vector(vector):
    serialized = pickle.dumps(vector)
    return serialized


def load_bbox_to_feature_tsv(tsv_path):
    bboxes_tsv = pd.read_csv(tsv_path, sep='\t')
    bbox_to_feature = {row['Name']: deserialize_feature_vector(row['Feature']) for index, row in bboxes_tsv.iterrows()}
    return bbox_to_feature


def simulate_bbox_to_feature_tsv():
    f_vactor = np.random.random((512, 1)).astype(np.float32)
    data = {'Name': ['thumb001_bbox_0'], 'Feature': [f_vactor]}
    bbox_to_feature_dataframe = pd.DataFrame.from_dict(data)
    tsv_path = r"C???\MockDetectedEmbeddings.tsv"
    bbox_to_feature_dataframe.to_csv(tsv_path, sep='\t')
    return


def load_features_dir(dir_path):
    files = os.listdir(dir_path)
    feature_files = [f for f in files if f.endswith('.fea')]
    feature_vector = {fea: np.load(fea) for fea in feature_files}
    return feature_vector
