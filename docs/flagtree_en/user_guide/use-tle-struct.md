# Use TLE-Struct

This section introduces how to use TLE-Struct. TLE-Struct is available on trition_3.6.x branch.

## GPU memory management

You can use the following operations to manage the GPU's memory.

### tle.gpu.alloc

The following example demonstrates how to reserve a block of memory in the GPU's high-speed on-chip  SMEM (Shared Memory) with dimensions `XBLOCK * YBLOCK` and data type `float32`.

```{code-block} python
a_smem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float32,
                      layout=None, scope=tle.gpu.storage_kind.smem)
```

### tle.gpu.local_ptr

Obtain the memory pointer.

```{code} python
# Get pointers to a_smem[0,:]: [(0, 0), (0, 1)...(0, YBLOCK-1)]
a_smem_ptrs = tle.gpu.local_ptr(a_smem,
    indices=(tl.broadcast(0, [YBLOCK]), tl.arrange(0, YBLOCK)))
```

- Signature: tle.local_ptr(buffer, indices) -> tl.tensor (pointer tensor)
- Purpose: Build arbitrary-shaped pointer views over shared memory buffer for tl.load/tl.store.
- Parameters:
  - buffer: buffered_tensor returned by tle.alloc (SMEM / TMEM).
  - indices: Tuple of integer tensors. Tuple length must equal rank(buffer), and every tensor must have identical shapes.
- Semantics:
  - Output pointer tensor shape equals the common shape of the indices tensors.
  - For each logical index (i0, i1, ...) in the output shape, the pointer value corresponds to buffer[indices0(i0, i1, ...), indices1(i0, i1, ...), ...].
  - Returned pointers live in shared memory address space (LLVM addrspace=3). Indices must be integers (i32/i64, etc., reduced to i32 during lowering).
  - Linearization is row-major (last dimension fastest); shared memory layout/encoding follows the buffer memdesc.

- Example 1: 1D slice

  ```{code-block} python
  smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)
  # Slice [offset, offset + SLICE)
  idx = offset + tl.arange(0, SLICE)
  slice_ptr = tle.local_ptr(smem, (idx,))
  vals = tl.load(slice_ptr)
  ```

- Example 2: K-dimension tiling (matrix slice)

  ```{code-block} python
  smem_a = tle.alloc([BM, BK], dtype=tl.float16, scope=tle.smem)
  # Slice (BM, KW), where KW is the K-dimension slice
  rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
  cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
  a_slice = tle.local_ptr(smem_a, (rows, cols))
  a_vals = tl.load(a_slice)
  ```
  
- Example 3: arbitrary gather view
  
  ```{code-block} python
      smem = tle.alloc([H, W], dtype=tl.float32, scope=tle.smem)
      # Take an offset column per row
      rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
      cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
      gather_ptr = tle.local_ptr(smem, (rows, cols))
      out = tl.load(gather_ptr)
  ```


### tle.gpu.copy

The following example demonstrates how to load a tile of data from the low-speed GMEM (Global Memory) into the high-speed on-chip SMEM.

- Copy from source:
  - `a_ptrs`: The base pointer(s) in GMEM
  - `ystride_a * yoffs[None, :]`: An offset vector added to the base pointer.
    - `yoffs[None, :]`: Represents a range of Y-axis offsets, broadcasted to a row vector.
    - `ystride_a`: The stride between rows in the source layout. This calculates the exact addresses of the 2D block that tends to load from GMEM.
- To destination:
  - `a_smem`: The previously allocated SMEM buffer. Data will be written here for fast access by the threads in this block.

```{code-block} python
tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

### tl.load for smem


### tl.store for smem


### tl.atomic_* for smem


## DSA memory management

### tle.dsa.alloc

Specific for allocating memory for Huawei Ascend.

```{code} python
a_ub = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32,
                      layout=tle.dsa.ascend.NZ, scope=tle.dsa.ascend.UB)
```

### tle.dsa.copy

Memory copy.

```{code-block} python
tle.dsa.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

