import numpy as np
import torch
from torch import nn

from second.pytorch.models import rpn
from second.pytorch.models.voxelnet import VoxelNet, register_voxelnet


class SmallObjectHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        final_num_filters = 64
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H,
                                   W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H,
                                   W).permute(0, 1, 3, 4, 2).contiguous()
        # print('small', box_preds.shape)

        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc,
                                               self._num_direction_bins, H,
                                               W).permute(0, 1, 3, 4,
                                                          2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(
                batch_size, -1, self._num_direction_bins)
        return ret_dict


class DefaultHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        final_num_filters = num_filters
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H,
                                   W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H,
                                   W).permute(0, 1, 3, 4, 2).contiguous()
        # print('large', box_preds.shape)

        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc,
                                               self._num_direction_bins, H,
                                               W).permute(0, 1, 3, 4,
                                                          2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(
                batch_size, -1, self._num_direction_bins)
        return ret_dict


@register_voxelnet
class VoxelNetNuscenesMultiHead(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)
        # print('self._num_class', self._num_class)

        small_classes = [
            c for c in ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
            if c in self.target_assigner.classes
        ]
        large_classes = [
            c for c in ["car", "truck", "trailer", "bus", "construction_vehicle"]
            if c in self.target_assigner.classes
        ]
        assert len(small_classes) > 0
        assert len(large_classes) > 0

        # print('num_anchors_per_location', self.target_assigner.num_anchors_per_location)
        # for c in self.target_assigner.classes:
        #     print(c, self.target_assigner.num_anchors_per_location_class(c))

        small_num_anchor_loc = sum([
            self.target_assigner.num_anchors_per_location_class(c)
            for c in small_classes
        ])
        large_num_anchor_loc = sum([
            self.target_assigner.num_anchors_per_location_class(c)
            for c in large_classes
        ])
        # print('small_num_anchor_loc', small_num_anchor_loc)
        # print('large_num_anchor_loc', large_num_anchor_loc)

        self.small_head = SmallObjectHead(
            num_filters=self.rpn._num_filters[0],
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )
        self.large_head = DefaultHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

    def network_forward(self, voxels, num_points, coors, batch_size):
        self.start_timer("forward.network.vfe")
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        self.end_timer("forward.network.vfe")

        self.start_timer("forward.network.mfe")
        spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size)
        self.end_timer("forward.network.mfe")

        self.start_timer("forward.network.rpn")
        rpn_out = self.rpn(spatial_features)
        r1 = rpn_out["stage0"]
        _, _, H, W = r1.shape
        # print('stage0.shape', r1.shape)
        cropsize40x40 = np.round(H * 0.1).astype(np.int64)
        r1 = r1[:, :, cropsize40x40:-cropsize40x40, cropsize40x40:-cropsize40x40]
        # print('cropped_stage0.shape', r1.shape)
        small = self.small_head(r1)
        large = self.large_head(rpn_out["out"])
        self.end_timer("forward.network.rpn")

        # concated preds MUST match order in class_settings in config.
        # print(large["box_preds"].shape, small["box_preds"].shape)
        res = {
            "box_preds": torch.cat([large["box_preds"], small["box_preds"]], dim=1),
            "cls_preds": torch.cat([large["cls_preds"], small["cls_preds"]], dim=1),
        }
        if self._use_direction_classifier:
            res["dir_cls_preds"] = torch.cat([large["dir_cls_preds"], small["dir_cls_preds"]], dim=1)
        return res
