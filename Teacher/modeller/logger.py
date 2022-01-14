from __future__ import print_function
import os
import logging
import time


class Logger(object):
    def __init__(self, root_output_path, logger_name):
        if not os.path.exists(root_output_path):
            os.makedirs(root_output_path)

        log_file = '{}_{}.log'.format(logger_name, time.strftime('%Y-%m-%d-%H-%M'))
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=os.path.join(root_output_path, log_file), format=head)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.rank = 0

    def info(self, msg):
        print(msg)
        if self.rank == 0:
            self.logger.info(msg)