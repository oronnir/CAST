import json
import os
from copy import deepcopy


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


class Configuration(Singleton):
    def __init__(self):
        self._configuration = None

    def get_configuration(self):
        """load the general configuration dictionary of Grouper"""
        if self._configuration is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
            with open(config_path, 'r') as conf_reader:
                self._configuration = json.load(conf_reader)

        return deepcopy(self._configuration)
