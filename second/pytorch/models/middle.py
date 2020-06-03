import numpy as np
import spconv
import torch
from torch import nn

from torchplus.nn import Empty
from torchplus.tools import change_default_args

REGISTERED_MIDDLE_CLASSES = {}


def register_middle(cls, name=None):
    global REGISTERED_MIDDLE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MIDDLE_CLASSES, f"exist class: {REGISTERED_MIDDLE_CLASSES}"
    REGISTERED_MIDDLE_CLASSES[name] = cls
    return cls


def get_middle_class(name):
    global REGISTERED_MIDDLE_CLASSES
    assert name in REGISTERED_MIDDLE_CLASSES, f"available class: {REGISTERED_MIDDLE_CLASSES}"
    return REGISTERED_MIDDLE_CLASSES[name]


@register_middle
class VoxelScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=(),
                 num_filters_down2=()):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        assert len(num_filters_down1) == 0
        assert len(num_filters_down2) == 0

        self.output_shape = output_shape
        self.nz = output_shape[1]
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels,
                                 self.nx * self.ny * self.nz,
                                 dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.nx * self.ny + \
                      this_coords[:, 2] * self.nx + \
                      this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels * self.nz, self.ny, self.nx)
        return batch_canvas


@register_middle
class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels,
                                 self.nx * self.ny,
                                 dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas


@register_middle
class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(
                spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm0"))
            middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(num_filters[-1],
                                num_filters[-1], (3, 1, 1), (2, 1, 1),
                                bias=False))
        middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(
                spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm1"))
            middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(num_filters[-1],
                                num_filters[-1], (3, 1, 1), (2, 1, 1),
                                bias=False))
        middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class SpMiddleFHDPeople(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHDPeople, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 21] -> [800, 600, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=[0, 1, 1]),  # [800, 600, 11] -> [400, 300, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [400, 300, 5] -> [400, 300, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class SpMiddle2K(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        super(SpMiddle2K, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(
                num_input_features, 8, 3,
                indice_key="subm0"),  # [3200, 2400, 81] -> [1600, 1200, 41]
            BatchNorm1d(8),
            nn.ReLU(),
            SubMConv3d(8, 8, 3, indice_key="subm0"),
            BatchNorm1d(8),
            nn.ReLU(),
            SpConv3d(8, 16, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm1"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm1"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm2"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm2"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 3
        self.grid = torch.full([self.max_batch_size, *sparse_shape],
                               -1,
                               dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size, self.grid)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class SpMiddleFHDLite(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDLite'):
        super(SpMiddleFHDLite, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 16, 3, 2,
                     padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)

        # ret.features = F.relu(ret.features)
        # print(self.middle_conv.fused())
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class SpMiddleFHDLiteHRZ(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDLite'):
        super(SpMiddleFHDLiteHRZ, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 32, 3, 2,
                     padding=1),  # [1600, 1200, 81] -> [800, 600, 41]
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 41] -> [400, 300, 21]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=1),  # [400, 300, 21] -> [200, 150, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class SpMiddleFHDHRZ(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHDHRZ, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3,
                                              momentum=0.01)(nn.BatchNorm1d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        else:
            BatchNorm1d = Empty
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2,
                     padding=1),  # [1600, 1200, 81] -> [800, 600, 41]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2,
                     padding=1),  # [800, 600, 41] -> [400, 300, 21]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2,
                     padding=1),  # [400, 300, 21] -> [200, 150, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


@register_middle
class Down4SparseMiddleNetwork(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=32,
                 num_filters_down1=(2, 2, 2),
                 num_filters_down2=(64, 128, 128)):
        super().__init__()
        # downsample_factor == 4
        assert len(num_filters_down1) == 3
        assert len(num_filters_down2) == 3

        # num_filters_down1: the number of each layers
        # num_filters_down2: the number of output channels of each layer

        self.name = 'SparseMiddleExtractor'
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SparseConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        else:
            BatchNorm1d = Empty
            SparseConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)

        sparse_shape = np.array(output_shape[1:4])
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape

        # the first layers
        num_filters = [num_input_features] + \
                      [num_filters_down2[0] for _ in range(num_filters_down1[0])]
        filters_pairs_d1 = [[
            num_filters[i], num_filters[i + 1]]
            for i in range(len(num_filters) - 1)]
        sub_spconv1, spconv1 = [], []
        for i, o in filters_pairs_d1:
            sub_spconv1.append(SubMConv3d(i, o, 3, indice_key="subm0"))
            sub_spconv1.append(BatchNorm1d(o))
            sub_spconv1.append(nn.ReLU())
        spconv1.append(
            SparseConv3d(
                in_channels=num_filters[-1],
                out_channels=num_filters[-1],
                kernel_size=3,
                stride=2,
                padding=1))
        spconv1.append(BatchNorm1d(num_filters[-1]))
        spconv1.append(nn.ReLU())

        # the second layers
        num_filters = [num_filters[-1]] + \
                      [num_filters_down2[1] for _ in range(num_filters_down1[1])]
        filters_pairs_d2 = [[
            num_filters[i], num_filters[i + 1]]
            for i in range(len(num_filters) - 1)]
        sub_spconv2, spconv2 = [], []
        for i, o in filters_pairs_d2:
            sub_spconv2.append(SubMConv3d(i, o, 3, indice_key="subm1"))
            sub_spconv2.append(BatchNorm1d(o))
            sub_spconv2.append(nn.ReLU())
        spconv2.append(
            SparseConv3d(
                in_channels=num_filters[-1],
                out_channels=num_filters[-1],
                kernel_size=3,
                stride=2,
                padding=1))
        spconv2.append(BatchNorm1d(num_filters[-1]))
        spconv2.append(nn.ReLU())

        # the third layers
        num_filters = [num_filters[-1]] + \
                      [num_filters_down2[2] for _ in range(num_filters_down1[2])]
        filters_pairs_d3 = [[
            num_filters[i], num_filters[i + 1]]
            for i in range(len(num_filters) - 1)]
        sub_spconv3, spconv3 = [], []
        for i, o in filters_pairs_d3:
            sub_spconv3.append(SubMConv3d(i, o, 3, indice_key="subm2"))
            sub_spconv3.append(BatchNorm1d(o))
            sub_spconv3.append(nn.ReLU())
        spconv3.append(
            SparseConv3d(
                in_channels=num_filters[-1],
                out_channels=num_filters[-1],
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1))
        spconv3.append(BatchNorm1d(num_filters[-1]))
        spconv3.append(nn.ReLU())

        self._sub_spconv1 = spconv.SparseSequential(*sub_spconv1)
        self._spconv1 = spconv.SparseSequential(*spconv1)
        self._sub_spconv2 = spconv.SparseSequential(*sub_spconv2)
        self._spconv2 = spconv.SparseSequential(*spconv2)
        self._sub_spconv3 = spconv.SparseSequential(*sub_spconv3)
        self._spconv3 = spconv.SparseSequential(*spconv3)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        sparse_feature = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        sparse_feature = self._sub_spconv1(sparse_feature)
        sparse_feature = self._spconv1(sparse_feature)  # [D, H, W] -> [D/2, H/2, W/2]
        sparse_feature = self._sub_spconv2(sparse_feature)
        sparse_feature = self._spconv2(sparse_feature)  # [D/2, H/2, W/2] -> [D/4, H/4, W/4]
        dense1 = sparse_feature.dense()
        N, C, D, H, W = dense1.shape
        dense1 = dense1.view(N, C * D, H, W)

        sparse_feature = self._sub_spconv3(sparse_feature)
        sparse_feature = self._spconv3(sparse_feature)  # [D/4, H/4, W/4] -> [D/8, H/8, W/8]
        dense2 = sparse_feature.dense()
        N, C, D, H, W = dense2.shape
        dense2 = dense2.view(N, C * D, H, W)

        return dense1, dense2
