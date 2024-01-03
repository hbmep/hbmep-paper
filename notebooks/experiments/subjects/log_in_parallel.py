import time
import logging

from joblib import Parallel, delayed


def start_logger_if_necessary():
    logger = logging.getLogger(__name__)
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(FORMAT))
        fh = logging.FileHandler('/home/vishu/logs/log.log', mode='w')
        fh.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


def calculate_square(num):
    logger = start_logger_if_necessary()
    time.sleep(1)   # Simulate a time-consuming task
    result = num ** 2
    logger.info(f"Square of {num} is {result}")
    print(f"Square of {num} is {result}")
    return result


def main():
    logger = start_logger_if_necessary()
    logger.info("Starting multiprocessing")
    with Parallel(n_jobs=4, backend="threading") as parallel:
        results = parallel(delayed(calculate_square)(num) for num in range(12))
    logger.info(f"Squares: {results}")
    return


if __name__ == "__main__":
    main()
