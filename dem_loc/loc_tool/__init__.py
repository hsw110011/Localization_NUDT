"""loc_tool Python 包 — localization_python.py 的依赖库。

模块:
    dem_tool         加载 DSM GeoTIFF（load_dem_tiff）
    coord_converter  WGS84 / 高斯 / 像素坐标转换
    cinterface       ROS 话题订阅（对应 C++ CInterface）
    common_struct    InputData 等数据结构（对应 C++ CommonStruct.h）

用法:
    from loc_tool import CoordConverter, InputData, load_dem_tiff
    from loc_tool.cinterface import CInterface
"""

from .common_struct import InputData
from .coord_converter import CoordConverter
from .dem_tool import DemTiffData, load_dem_tiff
from .dsm_patch import DsmPatchCropper, bev_grid_shape

__all__ = [
    "CInterface",
    "CoordConverter",
    "DemTiffData",
    "DsmPatchCropper",
    "InputData",
    "TOPIC_DEFINITIONS",
    "bev_grid_shape",
    "load_dem_tiff",
    "make_input_data",
]


def __getattr__(name):
    if name in ("CInterface", "TOPIC_DEFINITIONS", "make_input_data"):
        from .cinterface import CInterface, TOPIC_DEFINITIONS, make_input_data

        values = {
            "CInterface": CInterface,
            "TOPIC_DEFINITIONS": TOPIC_DEFINITIONS,
            "make_input_data": make_input_data,
        }
        globals().update(values)
        return values[name]
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
