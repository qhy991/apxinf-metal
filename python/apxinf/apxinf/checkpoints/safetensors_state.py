"""Read small processor-state SafeTensors files without torch.

Model weights stay in Rust.  LeRobot normalization sidecars are tiny host-side
metadata, so the checkpoint adapter reads only the numeric dtypes used for
processor state and returns numpy arrays.  This intentionally is not a second
model-weight loader.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict

import numpy as np


class SafeTensorStateError(ValueError):
    pass


_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def load_state_file(path) -> Dict[str, np.ndarray]:
    """Load a LeRobot processor state file into host numpy arrays."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SafeTensorStateError(f"read {path}: {exc}") from exc
    if len(raw) < 8:
        raise SafeTensorStateError(f"{path}: truncated SafeTensors header")
    header_len = struct.unpack("<Q", raw[:8])[0]
    if header_len > len(raw) - 8:
        raise SafeTensorStateError(
            f"{path}: header length {header_len} exceeds file size {len(raw)}"
        )
    try:
        header = json.loads(raw[8 : 8 + header_len])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeTensorStateError(f"{path}: invalid SafeTensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise SafeTensorStateError(f"{path}: SafeTensors header is not an object")

    data = memoryview(raw)[8 + header_len :]
    tensors: Dict[str, np.ndarray] = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            raise SafeTensorStateError(f"{path}: tensor {name!r} has no descriptor")
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype_name == "BF16":
            dtype = np.dtype("<u2")
        else:
            dtype = _DTYPES.get(dtype_name)
        if dtype is None:
            raise SafeTensorStateError(
                f"{path}: tensor {name!r} uses unsupported dtype {dtype_name!r}"
            )
        if (
            not isinstance(shape, list)
            or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise SafeTensorStateError(f"{path}: malformed descriptor for tensor {name!r}")
        start, end = offsets
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if start < 0 or end < start or end > len(data) or end - start != expected:
            raise SafeTensorStateError(
                f"{path}: invalid byte range {offsets} for tensor {name!r} shape {shape}"
            )
        array = np.frombuffer(data[start:end], dtype=dtype).reshape(shape)
        if dtype_name == "BF16":
            array = (array.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = np.array(array, copy=True)
    return tensors
