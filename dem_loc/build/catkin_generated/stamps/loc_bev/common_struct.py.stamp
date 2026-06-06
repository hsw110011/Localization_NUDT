#!/usr/bin/env python3
"""Python data containers matching src/include/CommonStruct.h."""

MAP_RESOLUTION = 0.15
PATH_RESOLUTION = 1.0
BASE_X = 19695752.27
BASE_Y = 3125228.07

ZONEWIDE_XG = 6.0001
ZONEWIDE = 6.0

DEG_TO_RAD = 3.141592653589793 / 180.0
RAD_TO_DEG = 180.0 / 3.141592653589793


class _Struct(object):
    __slots__ = ()

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        args = ", ".join(
            "{}={!r}".format(name, getattr(self, name)) for name in self.__slots__
        )
        return "{}({})".format(self.__class__.__name__, args)


class BLH_Point(_Struct):
    __slots__ = ("Lon", "Lat", "Height")

    def __init__(self, Lon=0.0, Lat=0.0, Height=0.0):
        self.Lon = Lon
        self.Lat = Lat
        self.Height = Height


class GaussPoint(_Struct):
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class double3D(_Struct):
    __slots__ = ("x", "y", "theta")

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta


class double2D(_Struct):
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class PointxyRGB(_Struct):
    __slots__ = ("x", "y", "z", "color")

    def __init__(self, x=0.0, y=0.0, z=0.0, color=None):
        self.x = x
        self.y = y
        self.z = z
        self.color = color


class Pose(_Struct):
    __slots__ = ("x", "y", "yaw")

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw


class Pointxyz(_Struct):
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class WORLD_POINT(_Struct):
    __slots__ = (
        "timeflag",
        "BLH",
        "gauss",
        "pixel",
        "heading",
        "distance",
        "std_var",
    )

    def __init__(
        self,
        timeflag=0.0,
        BLH=None,
        gauss=None,
        pixel=None,
        heading=0.0,
        distance=0.0,
        std_var=0.0,
    ):
        self.timeflag = timeflag
        self.BLH = BLH if BLH is not None else BLH_Point()
        self.gauss = gauss if gauss is not None else GaussPoint()
        self.pixel = pixel
        self.heading = heading
        self.distance = distance
        self.std_var = std_var


class Paras(_Struct):
    __slots__ = (
        "mapPath",
        "topLeftX",
        "topLeftY",
        "lowRightX",
        "lowRightY",
        "speedUpperLimits",
        "PathResource",
    )

    def __init__(
        self,
        mapPath="",
        topLeftX=0.0,
        topLeftY=0.0,
        lowRightX=0.0,
        lowRightY=0.0,
        speedUpperLimits=0,
        PathResource=0,
    ):
        self.mapPath = mapPath
        self.topLeftX = topLeftX
        self.topLeftY = topLeftY
        self.lowRightX = lowRightX
        self.lowRightY = lowRightY
        self.speedUpperLimits = speedUpperLimits
        self.PathResource = PathResource


class MAP_INFO(_Struct):
    __slots__ = (
        "left_top",
        "right_down",
        "gauss_x_resolution",
        "gauss_y_resolution",
        "BLH_lon_resolution",
        "BLH_lat_resolution",
        "rows",
        "cols",
    )

    def __init__(
        self,
        left_top=None,
        right_down=None,
        gauss_x_resolution=0.0,
        gauss_y_resolution=0.0,
        BLH_lon_resolution=0.0,
        BLH_lat_resolution=0.0,
        rows=0,
        cols=0,
    ):
        self.left_top = left_top if left_top is not None else WORLD_POINT()
        self.right_down = right_down if right_down is not None else WORLD_POINT()
        self.gauss_x_resolution = gauss_x_resolution
        self.gauss_y_resolution = gauss_y_resolution
        self.BLH_lon_resolution = BLH_lon_resolution
        self.BLH_lat_resolution = BLH_lat_resolution
        self.rows = rows
        self.cols = cols


class MATCH_STATE(_Struct):
    __slots__ = ("delta_x", "delta_y", "delta_theta", "scale", "score")

    def __init__(
        self,
        delta_x=0.0,
        delta_y=0.0,
        delta_theta=0.0,
        scale=0.0,
        score=0.0,
    ):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_theta = delta_theta
        self.scale = scale
        self.score = score


class MATCH_SITE(_Struct):
    __slots__ = (
        "path_center",
        "path_index",
        "vehicle_frozen_global",
        "vehicle_frozen_local",
        "vehicle_frozen_lidar",
        "vehicle_frozen_state",
        "match_state",
        "map_position",
        "match_time",
        "site_type",
        "std_var",
        "diff_pathl",
    )

    def __init__(self):
        self.path_center = WORLD_POINT()
        self.path_index = 0
        self.vehicle_frozen_global = WORLD_POINT()
        self.vehicle_frozen_local = None
        self.vehicle_frozen_lidar = None
        self.vehicle_frozen_state = WORLD_POINT()
        self.match_state = MATCH_STATE()
        self.map_position = WORLD_POINT()
        self.match_time = 0.0
        self.site_type = 0
        self.std_var = 0.0
        self.diff_pathl = WORLD_POINT()


INPUT_DATA_FIELDS = (
    "ColorMap",
    "SimilarityMap",
    "EntityMap",
    "TerrainMap",
    "SemanticMap",
    "GlobalPose",
    "LocalPose",
    "LidarLocalPose",
    "ReferencePath",
    "Odom",
    "BevMasks",
    "Inspvax",
    "NavsatOdom",
    "KittiGpsFix",
)

INPUT_REFRESH_FIELDS = (
    "GlobalPose_refreshflag",
    "LocalPose_refreshflag",
    "LidarLocalPose_refreshflag",
    "PointCloud_refreshflag",
    "SematicMap_refreshflag",
    "ColorMap_refreshflag",
    "SimilarityMap_refreshflag",
    "TerrainMap_refreshflag",
    "EntityMap_refreshflag",
    "ReferencePath_refreshflag",
    "Odom_refreshflag",
    "BevMasks_refreshflag",
    "Inspvax_refreshflag",
    "NavsatOdom_refreshflag",
    "KittiGpsFix_refreshflag",
)


class InputData(_Struct):
    """Container filled by CInterface.ConvertToLocalData."""

    __slots__ = INPUT_DATA_FIELDS + INPUT_REFRESH_FIELDS

    def __init__(self):
        for name in INPUT_DATA_FIELDS:
            setattr(self, name, None)
        for name in INPUT_REFRESH_FIELDS:
            setattr(self, name, False)
