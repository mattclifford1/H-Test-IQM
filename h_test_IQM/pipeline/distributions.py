import numpy as np
import multiprocessing
from tqdm import tqdm


def test_single(args):
    results = get_IQM_sensitivity(
        img=args[0], IQM=args[1], distortion=args[2], num_samples=10)
    return results

def get_all_scores(im_iter, IQM, distortion, num_samples=10, mp=False):

    # cannot use multiprocessing in with many IQMs currently...
    if mp == True:
        n_cpus = multiprocessing.cpu_count()
        args = [(img, IQM, distortion) for img in im_iter]
        with multiprocessing.Pool(processes=n_cpus) as pool:
            scores = list(tqdm(pool.imap_unordered(test_single, args),
                               total=len(im_iter), leave=False))

    else:
        scores = []
        for img in tqdm(im_iter):
            results = get_IQM_sensitivity(
                img, IQM=IQM, distortion=distortion, num_samples=10)
            scores.append(results)

    # convert all into one
    all_scores = []
    for results in scores:
        for sample in results:
            all_scores.append(sample)
    return all_scores




def get_IQM_sensitivity(img, IQM, distortion, num_samples=1000):
    # IQM sensitivity values from a given random distortion
    scores = []
    for _ in range(num_samples):
        scores.append(_get_score_single(img, IQM, distortion))
    # get rid of None values
    scores = [s for s in scores if s is not None]
    return scores


def _get_score_single(img, IQM, distortion):
    img_distorted = distortion(img)
    if img_distorted is not None:
        return IQM(img, img_distorted)
    return None