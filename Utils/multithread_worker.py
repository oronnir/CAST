import threading
import logging
import time
from tqdm import tqdm


class ThreadPool(object):
    def __init__(self):
        super(ThreadPool, self).__init__()
        self.active = []
        self.lock = threading.Lock()

    def make_active(self, name):
        with self.lock:
            self.active.append(name)
            logging.debug('Running: %s', self.active)

    def make_inactive(self, name):
        with self.lock:
            self.active.remove(name)
            logging.debug('Running: %s', self.active)


class Semaphore:
    def __init__(self, n_threads):
        self.n_threads = n_threads
        self._THREAD_LOCK = threading.Lock()
        self._GLOBAL_COUNTER = 0
        self._tabu = set()

    def _run_single(self, s, pool, do_work, kwargs, input_item, progress_bar):
        logging.debug('Waiting to join the pool')
        with s:
            name = threading.currentThread().getName()
            pool.make_active(name)
            try:
                do_work(input_item, **kwargs)
            except Exception as e:
                print(f'failed processing item: {input_item} with exception: {e}')
                raise e
            pool.make_inactive(name)
            with self._THREAD_LOCK:
                self._GLOBAL_COUNTER += 1
                progress_bar.update(1)

    def parallelize(self, items, do_work, do_work_kwargs, is_kv=False):
        s = threading.Semaphore(self.n_threads)
        pool = ThreadPool()
        p_bar = tqdm(total=len(items))
        while len(self._tabu) < len(items):
            for item in items:
                with self._THREAD_LOCK:
                    if item in self._tabu:
                        time.sleep(1)
                        continue
                    else:
                        self._tabu.add(item)
                if is_kv:
                    kv_args = do_work_kwargs[item]
                    t = threading.Thread(target=self._run_single, args=(s, pool, do_work, kv_args, item, p_bar))
                else:
                    t = threading.Thread(target=self._run_single, args=(s, pool, do_work, do_work_kwargs, item, p_bar))
                t.start()

        while len(pool.active) > 0:
            time.sleep(1)
        with self._THREAD_LOCK:
            self._GLOBAL_COUNTER = 0
            self._tabu = set()
