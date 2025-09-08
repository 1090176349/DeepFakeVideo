from tqdm import tqdm

def progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs)
