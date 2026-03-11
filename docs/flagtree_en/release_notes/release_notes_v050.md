# FlagTree 0.5.0 Release

- **Added Features**
  - Officially released TLE feature, supporting 16 core primitives for TLE-Lite and TLE-Struct:
    - TLE-Lite primitives: `tle.load`, `tle.device_mesh`, `tle.sharding`, `tle.shard_id`, `tle.remote`, `tle.distributed_barrier`
    - TLE-Struct primitives:
      - NVIDIA platform: `tle.gpu.alloc`, `tle.gpu.local_ptr`, `tle.gpu.copy`, `tl.load` (for shared memory), `tl.store` (for shared memory), `tl.atomic_*` (for shared memory)
      - DSA-based platforms: `tle.dsa.alloc`, `tle.dsa.copy`, `tle.dsa.extract_slice`, `tle.dsa.insert_slice`
    - Supported hardware platforms：NVIDIA, Huawei Ascend, and Tsingmicro
  - Officially released FLIR (FlagTree Intermediate Representation).
    - Supported 80 Triton language primitives with comprehensive operator coverage
    - Enabled shared compiler passes across backends
    - Supported hardware platforms: AIPU, Huawei Ascend, and Tsingmicro
