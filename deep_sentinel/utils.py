import logging
import os
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Callable, Any, List, Union

logger = logging.getLogger(__name__)


def to_path(p: 'Union[str, Path]') -> 'Path':
    if isinstance(p, Path):
        return p
    elif isinstance(p, str):
        return Path(p)
    else:
        raise TypeError(f"str or Path is expected, but actual {p.__class__.__name__}")


def to_absolute(p: 'Path') -> 'Path':
    return p.expanduser().resolve()


def exists(p: 'Path') -> bool:
    return to_absolute(p).exists()


def mkdir(p: 'Path') -> 'Path':
    p = to_absolute(p)
    if p.exists():
        if p.is_file():
            raise NotADirectoryError
    else:
        p.mkdir(parents=True)
    return p


def parallel(func: 'Callable[[Any], Any]', iterable: 'List[Any]', n_jobs: int = -1) -> 'List[Any]':
    if n_jobs < 0:
        n_jobs = os.cpu_count()
    if n_jobs > len(iterable):
        n_jobs = len(iterable)
    if n_jobs == 1:
        return [func(it) for it in iterable]
    with Pool(n_jobs) as pool:
        # executors = pool.imap(func, iterable)
        progress_bar = tqdm(total=len(iterable), leave=False)
        finished = 0
        results = [pool.apply_async(func, (it,)) for it in iterable]
        try:
            while True:
                time.sleep(0.5)
                statuses = [r.ready() for r in results]
                finished_now = statuses.count(True)
                progress_bar.update(finished_now - finished)
                finished = finished_now
                if all(statuses):
                    break
            results = [r.get(1) for r in results]
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            logger.error("SIGTERM Received. Shutdown workers.")
            pool.terminate()
            logger.error("Wait for shutting down...")
            pool.join()
            exit(1)
        finally:
            progress_bar.close()
    return results


def avoid_override(path: 'Path') -> 'Path':
    time_format = "_%Y-%m-%d_%H-%M-%S"
    new_file = path
    while new_file.exists():
        now = datetime.now()
        new_file = path.parent / (path.stem + now.strftime(time_format) + path.suffix)
    return new_file
