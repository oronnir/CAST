import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
from Animator.utils import eprint


class Semaphore:
    def __init__(self, n_processes=6):
        self.n_processes = min(n_processes, mp.cpu_count())

    def parallelize(self, items, do_work, timeout=24*3600):
        print(f'allocating {self.n_processes} processes to run {do_work.__name__} in parallel...')
        try:
            results = Parallel(n_jobs=self.n_processes, timeout=timeout, verbose=1)(delayed(do_work)(**parameters)
                                                                                    for parameters in tqdm(items))
            print(f'MP Semaphore is done running {do_work.__name__}!')
            return results
        except TimeoutError as toe:
            eprint(f"We lacked patience and got a multiprocessing.TimeoutError: {toe}")
            raise toe
        except Exception as e:
            eprint(f'MP parallelize has raised an exception: {e}')
            raise e
