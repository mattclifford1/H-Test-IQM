'''
Trivial, non-perceptual scorers. These exist as controls.

The project's claim is that a perceptual autoencoder's code occupancy captures something about
natural-image statistics. That claim is only interesting if it beats statistics that cost
nothing and know nothing about perception. If a JPEG byte count separates the same datasets
just as well, the paper has to say so -- see FINDINGS.md 5.1.4.
'''
import io
import numpy as np
import PIL.Image

from h_test_IQM.scorers.numpy_scorers import base_scorer_numpy


class pixel_std_model(base_scorer_numpy):
    '''Per-image standard deviation of pixel values. About as dumb as a statistic gets.'''
    def setup(self, **kwargs):
        pass

    def get_score(self, x):
        return float(np.std(x))


class pixel_entropy_model(base_scorer_numpy):
    '''Shannon entropy (bits) of the image's 256-bin grey-level histogram.'''
    def setup(self, bins=256, **kwargs):
        self.bins = bins

    def get_score(self, x):
        grey = x.mean(axis=2) if x.ndim == 3 else x
        counts, _ = np.histogram(grey, bins=self.bins, range=(0, 1))
        p = counts[counts > 0] / counts.sum()
        return float(-np.sum(p * np.log2(p)))


class jpeg_bytes_model(base_scorer_numpy):
    '''
    Compressed size in bytes per pixel, at fixed JPEG quality. A crude but genuinely
    competitive proxy for "how much structure is in this image" -- noise does not compress.
    '''
    def setup(self, quality=75, **kwargs):
        self.quality = quality

    def get_score(self, x):
        arr = np.clip(x, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 2:
            img = PIL.Image.fromarray(arr, mode='L')
        else:
            img = PIL.Image.fromarray(arr, mode='RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        return float(buf.getbuffer().nbytes) / (arr.shape[0] * arr.shape[1])


if __name__ == '__main__':
    # noise should be high-std / high-entropy / expensive to compress; a flat image the opposite
    noise = np.random.rand(4, 64, 64, 3).astype(np.float32)
    flat = np.ones((4, 64, 64, 3), dtype=np.float32) * 0.5
    for name, model in [('pixel_std', pixel_std_model(im_size=(64, 64))),
                        ('pixel_entropy', pixel_entropy_model(im_size=(64, 64))),
                        ('jpeg_bytes', jpeg_bytes_model(im_size=(64, 64)))]:
        print(f'{name:14s} noise={model(noise).mean():.4f}  flat={model(flat).mean():.4f}')
