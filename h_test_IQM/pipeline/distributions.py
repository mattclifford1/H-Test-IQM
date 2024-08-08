import numpy as np


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