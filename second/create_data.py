from pathlib import Path

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.lyft_dataset as lyft_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database


def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path,
                                Path(root_path) / "kitti_infos_train.pkl")


def nuscenes_data_prep(root_path,
                       version,
                       dataset_name='NuScenesDataset',
                       max_sweeps=10,
                       use_flat_vehicle_coords=False,
                       use_second_format_direction=False):
    assert not use_flat_vehicle_coords, 'use_flat_vehicle_coords is not supported anymore.'
    assert not use_second_format_direction, 'use_second_format_direction is not supported anymore.'

    # Note: replace 'splits' to 'train', 'val' or 'test'
    postfix = 'zero_slope' if use_flat_vehicle_coords else 'slope'
    filename_template = f'infos_splits_{postfix}_sweepsN.pkl'
    filename_template = filename_template.replace('N', str(max_sweeps))

    nu_ds.create_nuscenes_infos(root_path,
                                version=version,
                                filename_template=filename_template,
                                max_sweeps=max_sweeps,
                                use_flat_vehicle_coords=use_flat_vehicle_coords,
                                use_second_format_direction=use_second_format_direction)

    # filename_template = f'infos_train_slope_sweeps{max_sweeps}_12.pkl'
    split = 'test' if 'test' in version else 'train'
    name = filename_template.replace('splits', split)
    create_groundtruth_database(
        dataset_class_name=dataset_name,
        data_path=root_path,
        info_path=Path(root_path) / name,
        db_dirname=f'gt_database_sweeps{max_sweeps}',
    )


def lyft_data_prep(root_path,
                   version='v1.01-train',
                   use_flat_vehicle_coordinates=True,
                   use_second_format_direction=False):
    lyft_ds.create_lyft_infos(
        root_path,
        version,
        use_flat_vehicle_coordinates,
        use_second_format_direction
    )


if __name__ == '__main__':
    fire.Fire()
