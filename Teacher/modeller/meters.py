import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter:

    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()
