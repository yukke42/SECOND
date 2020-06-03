# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/similarity.proto

import sys

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_b = sys.version_info[0] < 3 and (lambda x: x) or (
    lambda x: x.encode('latin1'))
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='second/protos/similarity.proto',
    package='second.protos',
    syntax='proto3',
    serialized_options=None,
    serialized_pb=_b(
        '\n\x1esecond/protos/similarity.proto\x12\rsecond.protos\"\xff\x01\n\x1aRegionSimilarityCalculator\x12\x43\n\x15rotate_iou_similarity\x18\x01 \x01(\x0b\x32\".second.protos.RotateIouSimilarityH\x00\x12\x45\n\x16nearest_iou_similarity\x18\x02 \x01(\x0b\x32#.second.protos.NearestIouSimilarityH\x00\x12@\n\x13\x64istance_similarity\x18\x03 \x01(\x0b\x32!.second.protos.DistanceSimilarityH\x00\x42\x13\n\x11region_similarity\"\x15\n\x13RotateIouSimilarity\"\x16\n\x14NearestIouSimilarity\"Z\n\x12\x44istanceSimilarity\x12\x15\n\rdistance_norm\x18\x01 \x01(\x02\x12\x15\n\rwith_rotation\x18\x02 \x01(\x08\x12\x16\n\x0erotation_alpha\x18\x03 \x01(\x02\x62\x06proto3'
    ))

_REGIONSIMILARITYCALCULATOR = _descriptor.Descriptor(
    name='RegionSimilarityCalculator',
    full_name='second.protos.RegionSimilarityCalculator',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='rotate_iou_similarity',
            full_name=
            'second.protos.RegionSimilarityCalculator.rotate_iou_similarity',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='nearest_iou_similarity',
            full_name=
            'second.protos.RegionSimilarityCalculator.nearest_iou_similarity',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='distance_similarity',
            full_name=
            'second.protos.RegionSimilarityCalculator.distance_similarity',
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name='region_similarity',
            full_name=
            'second.protos.RegionSimilarityCalculator.region_similarity',
            index=0,
            containing_type=None,
            fields=[]),
    ],
    serialized_start=50,
    serialized_end=305,
)

_ROTATEIOUSIMILARITY = _descriptor.Descriptor(
    name='RotateIouSimilarity',
    full_name='second.protos.RotateIouSimilarity',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=307,
    serialized_end=328,
)

_NEARESTIOUSIMILARITY = _descriptor.Descriptor(
    name='NearestIouSimilarity',
    full_name='second.protos.NearestIouSimilarity',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=330,
    serialized_end=352,
)

_DISTANCESIMILARITY = _descriptor.Descriptor(
    name='DistanceSimilarity',
    full_name='second.protos.DistanceSimilarity',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='distance_norm',
            full_name='second.protos.DistanceSimilarity.distance_norm',
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='with_rotation',
            full_name='second.protos.DistanceSimilarity.with_rotation',
            index=1,
            number=2,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='rotation_alpha',
            full_name='second.protos.DistanceSimilarity.rotation_alpha',
            index=2,
            number=3,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=354,
    serialized_end=444,
)

_REGIONSIMILARITYCALCULATOR.fields_by_name[
    'rotate_iou_similarity'].message_type = _ROTATEIOUSIMILARITY
_REGIONSIMILARITYCALCULATOR.fields_by_name[
    'nearest_iou_similarity'].message_type = _NEARESTIOUSIMILARITY
_REGIONSIMILARITYCALCULATOR.fields_by_name[
    'distance_similarity'].message_type = _DISTANCESIMILARITY
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
    _REGIONSIMILARITYCALCULATOR.fields_by_name['rotate_iou_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name[
    'rotate_iou_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name[
        'region_similarity']
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
    _REGIONSIMILARITYCALCULATOR.fields_by_name['nearest_iou_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name[
    'nearest_iou_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name[
        'region_similarity']
_REGIONSIMILARITYCALCULATOR.oneofs_by_name['region_similarity'].fields.append(
    _REGIONSIMILARITYCALCULATOR.fields_by_name['distance_similarity'])
_REGIONSIMILARITYCALCULATOR.fields_by_name[
    'distance_similarity'].containing_oneof = _REGIONSIMILARITYCALCULATOR.oneofs_by_name[
        'region_similarity']
DESCRIPTOR.message_types_by_name[
    'RegionSimilarityCalculator'] = _REGIONSIMILARITYCALCULATOR
DESCRIPTOR.message_types_by_name['RotateIouSimilarity'] = _ROTATEIOUSIMILARITY
DESCRIPTOR.message_types_by_name[
    'NearestIouSimilarity'] = _NEARESTIOUSIMILARITY
DESCRIPTOR.message_types_by_name['DistanceSimilarity'] = _DISTANCESIMILARITY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegionSimilarityCalculator = _reflection.GeneratedProtocolMessageType(
    'RegionSimilarityCalculator',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_REGIONSIMILARITYCALCULATOR,
        __module__='second.protos.similarity_pb2'
        # @@protoc_insertion_point(class_scope:second.protos.RegionSimilarityCalculator)
    ))
_sym_db.RegisterMessage(RegionSimilarityCalculator)

RotateIouSimilarity = _reflection.GeneratedProtocolMessageType(
    'RotateIouSimilarity',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_ROTATEIOUSIMILARITY,
        __module__='second.protos.similarity_pb2'
        # @@protoc_insertion_point(class_scope:second.protos.RotateIouSimilarity)
    ))
_sym_db.RegisterMessage(RotateIouSimilarity)

NearestIouSimilarity = _reflection.GeneratedProtocolMessageType(
    'NearestIouSimilarity',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_NEARESTIOUSIMILARITY,
        __module__='second.protos.similarity_pb2'
        # @@protoc_insertion_point(class_scope:second.protos.NearestIouSimilarity)
    ))
_sym_db.RegisterMessage(NearestIouSimilarity)

DistanceSimilarity = _reflection.GeneratedProtocolMessageType(
    'DistanceSimilarity',
    (_message.Message, ),
    dict(
        DESCRIPTOR=_DISTANCESIMILARITY,
        __module__='second.protos.similarity_pb2'
        # @@protoc_insertion_point(class_scope:second.protos.DistanceSimilarity)
    ))
_sym_db.RegisterMessage(DistanceSimilarity)

# @@protoc_insertion_point(module_scope)
