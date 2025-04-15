from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """
    Theoretical Numbers from data sheet. If two numbers are given, Tensor/Matrix Core vs not,
    then the higher number is reported. Sparsity is not considered.


    Bandwidth numbers are tricky, because there are platform differences that may not show up in the profiler trace.
    For example,
    """

    tops: dict[torch.dtype, float]
    dram_bw_tbs: float
    dram_gb: float


# TODO investigate profiler support for tf32 and allow device to report correct number when it's turned on.
_device_mapping: dict[str, DeviceInfo] = {
    # Source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    "NVIDIA H100": DeviceInfo(
        tops={
            torch.float64: 9.7,
            torch.float32: 19.5,
            torch.bfloat16: 1979.0,
            torch.float16: 1979.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.float8_e4m3fnuz: 3958.0,
            torch.float8_e5m2: 3958.0,
            torch.float8_e5m2fnuz: 3958.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.int8: 3958.0,
        },
        dram_bw_tbs=3350,
        dram_gb=80,
    ),
    # Source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    "NVIDIA A100": DeviceInfo(
        tops={
            torch.float64: 19.5,
            torch.float32: 19.5,
            torch.bfloat16: 312.5,
            torch.float16: 312.5,
            # Not in datasheet: float8
            torch.int8: 624.0,
        },
        dram_bw_tbs=2039.0,
        dram_gb=80.0,
    ),
    # Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf
    "AMD MI300A": DeviceInfo(
        tops={
            torch.float64: 122.6,
            torch.float32: 122.6,
            # torch.tf32: 490.3,
            torch.bfloat16: 980.6,
            torch.float16: 980.6,
            torch.float8_e8m0fnu: 1961.2,
            torch.float8_e8m0fnu: 1961.2,
            torch.float8_e4m3fnuz: 1961.2,
            torch.float8_e5m2: 1961.2,
            torch.float8_e5m2fnuz: 1961.2,
            torch.float8_e8m0fnu: 1961.2,
            torch.int8: 1961.2,
        },
        dram_bw_tbs=5300.0,
        dram_gb=128.0,
    ),
    # Source: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    "AMD MI300X": DeviceInfo(
        tops={
            torch.float64: 163.4,
            torch.float32: 163.4,
            torch.bfloat16: 1307.4,
            torch.float16: 1307.4,
            torch.float8_e8m0fnu: 2614.9,
            torch.float8_e8m0fnu: 2614.9,
            torch.float8_e4m3fnuz: 2614.9,
            torch.float8_e5m2: 2614.9,
            torch.float8_e5m2fnuz: 2614.9,
            torch.float8_e8m0fnu: 2614.9,
            torch.int8: 2614.9,
        },
        dram_bw_tbs=5300.0,
        dram_gb=192.0,
    ),
}


def lookup_device_info(name: str) -> "DeviceInfo":
    """
    Problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device. Therefore, _device_mapping statically contains the information for lots of devices.
    If one is missing, please run DeviceInfo.get_device_info() and add it to _device_mapping.
      name (str): name of the device to lookup. Should map onto torch.cuda.get_device_name().
    """
    if name not in _device_mapping:
        raise RuntimeError(
            f"Unsupported device in profile: {name}, please consider contributing to _device_mapping."
        )
    return _device_mapping[name]


def datasheet_tops(dtype: torch.dtype) -> float:
    """
    Get the theoretical TFLOPS of the device for a given dtype. This can throw an exception if the device
    is not in the datasheet list above.
    """
    name = torch.cuda.get_device_name()
    device_info = lookup_device_info(name)
    return device_info.tops[dtype]
