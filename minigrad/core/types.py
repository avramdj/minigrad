from typing import Dict, Tuple, Type, TypeVar

import numpy as np

_ctype_map: Dict[Type, str] = {
    # np.float16: "half",
    np.float32: "float",
    np.float64: "double",
    np.int8: "char",
    np.int16: "short",
    np.int32: "int",
    np.int64: "long",
}

float32 = np.float32
float64 = np.float64
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

MinigradType = TypeVar("MinigradType", float32, float64, int8, int16, int32, int64)
