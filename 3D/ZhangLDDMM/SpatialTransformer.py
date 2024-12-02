import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        x, y, z = [1/s*torch.arange(0, s, dtype=torch.float32) for s in size]
        self.X, self.Y, self.Z = torch.meshgrid(x, y, z, indexing='xy')
        # grid = torch.stack(self.X, self.Y, self.Z)
        # print('grid shape init:', grid.shape)
        # grid = torch.unsqueeze(grid, 0)

        # print('grid shape init:', grid.shape)
        # grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)

    def forward(self, src, flowX, flowY, flowZ, return_phi=False):
        self.X = self.X.to(flowX.device)
        self.Y = self.Y.to(flowY.device)
        self.Z = self.Z.to(flowZ.device)

        # new locations
        new_X = self.X + flowX
        new_Y = self.Y + flowY
        new_Z = self.Z + flowZ

        # Normalizar newX entre [-1, 1]
        maxX = torch.max(new_X)
        minX = torch.min(new_X)
        new_X = 2 * (new_X - minX) / (maxX - minX) - 1

        # Normalizar newY entre [-1, 1]
        maxY = torch.max(new_Y)
        minY = torch.min(new_Y)
        new_Y = 2 * (new_Y - minY) / (maxY - minY) - 1

        # Normalizar newZ entre [-1, 1]
        maxZ = torch.max(new_Z)
        minZ = torch.min(new_Z)
        new_Z = 2 * (new_Z - minZ) / (maxZ - minZ) - 1

        # Comprobación de seguridad: nada esta fuera de los límites
        assert torch.max(new_X) <= 1 and torch.min(new_X) >= -1, 'Pos X in interpolation out of bounds'
        assert torch.max(new_Y) <= 1 and torch.min(new_Y) >= -1, 'Pos Y in interpolation out of bounds'
        assert torch.max(new_Z) <= 1 and torch.min(new_Z) >= -1, 'Pos Z in interpolation out of bounds'

        new_locs = torch.stack((new_X, new_Y, new_Z))

        new_locs = new_locs.unsqueeze(0)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 0, 1]]

        # Eliminar variables de gpu
        del new_X, new_Y, new_Z

        if return_phi:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), new_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)