import torch

UNIFORM_INSTANCES = 60000

class UNIFORM_LOADER:
    def __init__(self, normalise=(0, 1), length=UNIFORM_INSTANCES, size=(3, 32, 32),
                 dtype=torch.float32,
                 device='cpu',
                 indicies_to_use='all',
                 **kwargs):
        self.normalise = normalise
        self.length = length
        self.size = size
        self.dtype = dtype
        self.device = device

        self.reduce_data(indicies_to_use)

    def _process_image(self, image):
        image = image*(self.normalise[1]-self.normalise[0])
        image = image + self.normalise[0]
        return image

    def _get_image(self):
        image_raw = torch.rand(self.size, dtype=self.dtype, device=self.device)
        image = self._process_image(image_raw)
        return image
    
    def reduce_data(self, indicies_to_use):
        if indicies_to_use == 'all':
            return
        else:
            self.length = len(indicies_to_use)

    
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        # get image
        image = self._get_image()
        return image, 0, 0
    

