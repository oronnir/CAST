import functools
import io
import shutil
import sys
import time
import traceback
import socket
import json
import pickle
import os

import base64
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import ParameterGrid
from collections import Iterable, OrderedDict, defaultdict
import termcolor
import re
import copy
import colorama
colorama.init()

camel_pat = re.compile(r'([A-Z])')
under_pat = re.compile(r'_([a-z])')
FEATURE_VEC = ['clustering_method', 'normalization_method']
HOST_NAME = socket.gethostname()
hostName = None
EPS = 1e-5

# read configuration
config_path = os.path.join(os.path.dirname(__file__), '', 'Configuration.json')
if os.path.isfile(config_path):
    with open(config_path) as data_file:
        data = json.load(data_file)
        locals().update(data)


def eprint(message, exception=None):
    """print error message to the stderr"""
    sys.stderr.write(str(message))
    if exception is not None:
        sys.stderr.write(str(exception))
        tb = traceback.format_exc()
        sys.stderr.write(tb)
    traceback.print_exc(file=sys.stderr)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2bool_true_or_false(v):
    return v in (False, True)


def colored(message, color):
    if hostName == HOST_NAME:
        return termcolor.colored(message, color)
    return message


def profiling(item_finished, prev_timestamp):
    now = time.time()
    print(colored("-PROFILING- Duration: {:.3f}sec. Task: {}".format(now - prev_timestamp, item_finished), 'blue'))
    return now


def trace_ordered(values_dict):
    return ", ".join(FEATURE_VEC[i] + "={" + str(values_dict[FEATURE_VEC[i]]) + "}" for i in range(len(FEATURE_VEC)))


def grid_search(input_dir, train_model, hyper_parameters, stats_output):
    """
    Grid Search util method to go over all possible configurations and train a model
    :param stats_output: output file path with all statistics for all iterations stats aggregations
    :param train_model: model trainer method per hyper parameters configuration
    :param hyper_parameters: a list of tuples where the first element is the name of the hyper-parameter and the second
    is a list per hyper-parameter possible values
    example: param_grid=[('l1_regularization', [0.01, 0.1]), ('gradient_clipping', [100.0, 1000.0])]
    :return: list of train_model results
    """
    param_dict = dict(hyper_parameters)
    grid = ParameterGrid(param_dict)
    iteration = 1
    total_iterations = functools.reduce(lambda x, y: x * y, [len(dim) for dim in param_dict.values()])
    results = []

    for params in grid:
        config = trace_ordered(params)
        print('******* Starting new session #{0} out of: {1} *******'.format(iteration, total_iterations))
        print('Configuration: ' + config)
        start = time.time()
        evaluation_metrics = None

        try:
            evaluation_metrics = train_model(input_dir, params, iteration)

        except Exception as e:
            tb = traceback.format_exc()
            eprint("\nConfiguration failed for config: " + config + '\n')
            eprint('The Exception thrown: \n' + str(e) + '\n')
            eprint('The stack trace: \n' + str(tb) + '\n')

        if evaluation_metrics is not None:
            results.append(evaluation_metrics)
            print("\nConfiguration succeeded for: {0} at {1:.2f} seconds. Estimated time to end: {2:.2f} seconds.\n"
                  "Eval metrics={3}"
                  .format(config, time.time() - start, (time.time() - start) * (total_iterations - iteration), evaluation_metrics))
            evaluation_metrics['iter'] = iteration
            if os.path.isfile(stats_output):
                evaluation_metrics.to_csv(stats_output, mode='a', header=False)
            else:
                evaluation_metrics.to_csv(stats_output)
        iteration += 1
    return results


def flatten(items):
    """ Yield items from any nested iterable """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def index_to_val_dict(list_of_values):
    count = 0
    index_to_value = {}
    for value in list_of_values:
        index_to_value[count] = value
        count += 1
    return index_to_value


def value_to_inx_dict(list_of_values):
    count = 0
    value_to_inx = {}
    for value in list_of_values:
        value_to_inx[value] = count
        count += 1
    return value_to_inx


def convert_to_dict(obj):
    """
    A function takes in a custom object and returns a dictionary representation of the object.
    This dict representation includes meta data such as the object's module and class names.
    """

    #  Populate the dictionary with object meta data
    obj_dict = {}

    #  Populate the dictionary with object properties
    obj_dict.update(obj.__dict__)

    return obj_dict


def dict_to_obj(our_dict, class_name, module_name):
    """
    Function that takes in a dict and returns a custom object associated with the dict.
    This function makes use of the "__module__" and "__class__" metadata in the dictionary
    to know which object type to create.
    """
    # We use the built in __import__ function since the module name is not yet known at runtime
    module_name = module_name.split('.')[-1]
    module = __import__(module_name)

    # Get the class from the module
    class_ = getattr(module, class_name)

    # Use dictionary unpacking to initialize the object
    obj = class_(our_dict)
    return obj


def hash_values(input_list):
    """
    maps/codes a list into integers
    :param input_list: enumerable of hashable elements
    """
    hash_map = dict()
    for val in input_list:
        hash_map[val] = hash_map.get(val, len(hash_map))
    return [hash_map[val] for val in input_list]


def underscore_to_camel(name):
    return under_pat.sub(lambda x: x.group(1).upper(), name)


def camel_to_underscore(name):
    return camel_pat.sub(lambda x: '_' + x.group(1).lower(), name)


def to_json(obj, json_path):
    """save as json or raise exception"""
    def get_camelcased_dict(some_dict):
        camel_cased_dict = dict()
        for k, v in some_dict.items():
            key = underscore_to_camel(k)
            key = key.lower() if len(key) == 1 else key[0].lower() + key[1:]
            camel_cased_dict[key] = v
        return camel_cased_dict

    try:
        clone = copy.deepcopy(obj)
        with open(json_path, 'w') as out_f:
            out_f.write(json.dumps(clone, default=lambda o: get_camelcased_dict(o.__dict__), sort_keys=True, indent=4))
        return
    except Exception as e:
        traceback.print_exc()
        eprint(' with exception: \'{}\'' % e)
        raise e


def create_dir_if_not_exist(parent, dir_name):
    """creates a folder dir_name under parent if is does not exist already - thread safe"""
    dir_path = os.path.join(parent, dir_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


def recreate_dir(parent, dir_name):
    """deletes dir if exist and creates a folder dir_name under parent - thread safe"""
    dir_path = os.path.join(parent, dir_name)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def sort_ordered_dict_by_key(ordered_dict: OrderedDict) -> OrderedDict:
    ordered_dict = OrderedDict(sorted(ordered_dict.items()))
    return ordered_dict


def deserialize_pickle(path_to_pickle):
    """load a pickle file"""
    pickle_reader = open(path_to_pickle, 'rb')
    vid_mot_assignment = pickle.load(pickle_reader)

    # close the file
    pickle_reader.close()
    return vid_mot_assignment


def serialize_pickle(output_path, obj):
    pickle_writer = open(output_path, 'wb')
    pickle.dump(obj, pickle_writer)
    pickle_writer.close()


def string_to_ndarray(base64_string):
    image = string_to_pil(base64_string)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def string_to_pil(base64_string):
    img_data = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(img_data))
    return image


def try_zip_folder(input_dir, output_zip):
    """zips an input folder into the output/path/filename.zip"""
    has_succeeded = False
    cwd = os.getcwd()
    try:
        parent_dir, input_dir_name = os.path.split(input_dir)
        os.chdir(parent_dir)
        zip_name_no_ext = output_zip.rsplit('.zip')[0]
        shutil.make_archive(base_name=zip_name_no_ext, format='zip')
        print('Triplets repo is zipped!')
        has_succeeded = os.path.isfile(output_zip)
    except Exception as e:
        eprint(f'Failed zipping folder: "{input_dir}" with exception: ', e)
    os.chdir(cwd)
    return has_succeeded
