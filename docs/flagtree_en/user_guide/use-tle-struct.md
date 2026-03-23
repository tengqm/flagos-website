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

- Signature: `tle.gpu.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: Build arbitrary-shaped pointer views over shared memory buffer for `tl.load`/`tl.store`.
- Parameters:
  - `buffer`: buffered_tensor returned by `tle.gpu.alloc` (SMEM / TMEM).
  - `indices`: An optional tuple of integer tensors, whose length must equal `rank(buffer)`, and each tensor must have the same shape. If omitted or passed as `None`, the backend will handle it according to full indices semantics.
- Semantics:
  - When `indices` are explicitly provided, the output pointer tensor has a shape equal to the common (broadcasted) shape of the indices.
  - For each logical index `(i0, i1, ...)` in the output shape, the corresponding pointer refers to `buffer[indices0(i0, ...), indices1(i0, ...), ...]`.
  - When `indices=None`, a full-view pointer covering the entire `buffer` is returned:
    - If rank > 0, a pointer tensor with shape equal to `shape(buffer)` is returned.
    - If rank = 0, a scalar pointer is returned.
  - The returned pointers reside in the shared memory address space (LLVM address space 3). Indices must be of integer type (e.g., i32, i64, etc.), and will be normalized to i32 during lowering.
  - Memory layout is linearized in row-major order (with the last dimension varying fastest). The shared memory layout and encoding follow the buffer's memdesc.

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

Supported downstream operations:

- `tl.load`
- `tl.store`
- `tl.atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_xchg`, `atomic_xor`

Practical notes:

- The availability of atomic operations depends on the element data type (dtype) and the capabilities of the backend hardware. It is recommended to prioritize integer or floating-point types that are explicitly verified as supported on the target hardware.

- For load-after-store hazards involving local_ptr, the TLE backend pass `TleInsertLocalPointerBarriers` automatically inserts necessary memory barriers. Manual barrier insertion is only required when using custom synchronization patterns that fall outside the scope of this pass.

- Example 4: Performing load, store, and atomic operations on the same local_ptr.

```{code-block} python
smem_i32 = tle.gpu.alloc([BLOCK], dtype=tl.int32, scope=tle.gpu.smem)
ptr = tle.gpu.local_ptr(smem_i32, (tl.arange(0, BLOCK),))

tl.store(ptr, tl.zeros([BLOCK], dtype=tl.int32))
tl.atomic_add(ptr, 1)
vals = tl.load(ptr)
```

### tle.gpu.local_ptr (for remote)

- Signature: `tle.gpu.local_ptr(remote_buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: Constructs a pointer view into a remote shared/local buffer returned by `tle.remote(...)`.
- Inputs:
  - `remote_buffer`: Returned by `tle.remote(buffer, shard_id, scope)`, where `buffer` is typically allocated via `tle.gpu.alloc`.
  - `indices`: Consistent with the local pattern (`None` denotes a full-view, or a tuple of integer tensors with matching shapes may be provided).
- Semantics:
  - The pointer’s shape, indexing behavior, and linearization rules are identical to those of the local `tle.gpu.local_ptr`.
  - Address resolution is routed to the remote shard specified by `shard_id`.
  - For cross-shard reads/writes that require ordering guarantees, use `tle.distributed_barrier(...)` in conjunction.
  
Read the remote SMEM tile on the neighboring shard.

```{code-block} python
smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.storage_kind.smem)
remote_smem = tle.remote(smem, shard_id=(node_rank, next_device), scope=mesh)

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
remote_ptr = tle.gpu.local_ptr(remote_smem, (rows, cols))

vals = tl.load(remote_ptr)
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

## DSA memory management and data movement

### tle.dsa.alloc

Signature: `tle.dsa.alloc(shape, dtype, mem_addr_space)`
Purpose: Allocates a DSA local buffer in the specified memory address space.
Address spaces exposed by Huawei Ascend:

- `tle.dsa.ascend.UB`
- `tle.dsa.ascend.L1`
- `tle.dsa.ascend.L0A`
- `tle.dsa.ascend.L0B`
- `tle.dsa.ascend.L0C`

```{code-block} python
a_ub = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
b_l1 = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.L1)
```

### tle.dsa.copy

Signature: `tle.dsa.copy(src, dst, shape, inter_no_alias=False)`
Purpose: Performs explicit data movement (bidirectional) between GMEM pointers and DSA local buffers.

```{code-block} python
tle.dsa.copy(x_ptrs, a_ub, [tail_m, tail_n])    # GMEM → local buffer  
tle.dsa.copy(a_ub, out_ptrs, [tail_m, tail_n])  # local buffer → GMEM
```

### tle.dsa.local_ptr  

- Signature: `tle.dsa.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`  
- Purpose: Constructs a pointer view over a DSA local buffer (e.g., UB or L1) to enable explicit local memory access patterns.  

- Parameters:  
  - `buffer`: A DSA-buffered tensor, typically allocated via `tle.dsa.alloc`.  
  - `indices`: Optional tuple of integer tensors; if omitted or set to `None`, the full index space is used (full-view semantics).  

Semantics:  
  The pointer view model is identical to that of `tle.gpu.local_ptr` (same shape and indexing rules).  
  Intended for DSA-local access patterns where explicit pointer materialization is required.  

```{code-block} python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_ptr = tle.dsa.local_ptr(a_ub, (rows, cols))
a_val = tl.load(a_ptr)
```

### tle.dsa.local_ptr (for remote)  

- Signature: `tle.dsa.local_ptr(remote_buffer, indices=None) -> tl.tensor | tl.ptr`  
- Purpose: Constructs a pointer view into a remote DSA local buffer returned by `tle.remote(...)`.  

- Inputs:  
  - `remote_buffer`: Returned by `tle.remote(dsa_buffer, shard_id, scope)`.  
  - `indices`: Same semantics as in the local DSA case.  `

- Semantics:  
  - Maintains the same pointer view rules as the local DSA variant.  
  - Dereferencing the pointer routes memory accesses to the remote shard identified by `shard_id`.  
  - When ordering across shards is required, use in conjunction with `tle.distributed_barrier(...)`.  

```{code-block} python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
remote_a_ub = tle.remote(a_ub, shard_id=peer_rank, scope=mesh)

rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
remote_ptr = tle.dsa.local_ptr(remote_a_ub, (rows, cols))
remote_val = tl.load(remote_ptr)
```

### tle.dsa.to_tensor and tle.dsa.to_buffer

- `tle.dsa.to_tensor(buffer, writable=True)`: Converts a DSA buffer into a tensor view to participate in tensor expressions.
- `tle.dsa.to_buffer(tensor, space)`: Converts tensor values back into a DSA buffer in the specified address space.

```{code-block} python
c_val = tle.dsa.to_tensor(c_ub, writable=True)
result = c_val * 0.5
d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
tle.dsa.copy(d_ub, out_ptrs, [tail_m, tail_n])
```

## Vector operators (Buffer form)

### tle.dsa.add, tle.dsa.sub, tle.dsa.mul, tle.dsa.div，tle.dsa.max，and tle.dsa.min

Built-in operators:
`tle.dsa.add`
`tle.dsa.sub`
`tle.dsa.mul`
`tle.dsa.div`
`tle.dsa.max`
`tle.dsa.min`

General signature:  
`tle.dsa.(lhs, rhs, out)`

Computation model:  
Performs element-wise binary operations on DSA-local buffers.

Shape rules:

- The rank and shape of `lhs`, `rhs`, and `out` must be identical.
- Implicit broadcasting is not performed by default at this API layer.

- Type rules:
  - In practice, it is recommended that all three operands use the same data type (`dtype`).
  - Integer types are typically used in indexing/counting paths, while floating-point types are commonly used in activation/numerical computation paths.

- Address space rules:
  - Buffers must be allocated in a DSA-local address space supported by the backend (e.g., UB/L1 combination).
  - Hot data should remain in local memory as much as possible to avoid unnecessary round trips to global memory (GMEM).

Operator semantics:
`tle.dsa.add(lhs, rhs, out)`: `out = lhs + rhs`
`tle.dsa.sub(lhs, rhs, out)`: `out = lhs - rhs`
`tle.dsa.mul(lhs, rhs, out)`: `out = lhs * rhs`
`tle.dsa.div(lhs, rhs, out)`: `out = lhs / rhs` (precision and rounding behavior depend on backend implementation)
`tle.dsa.max(lhs, rhs, out)`: `out = max(lhs, rhs)`
`tle.dsa.min(lhs, rhs, out)`: `out = min(lhs, rhs)`

In-place/reuse recommendations:

- Output buffers can be reused across multiple computation steps, e.g., `tle.dsa.mul(tmp, b, tmp)`.
- Unless the backend explicitly guarantees alias safety, input and output buffers should not share memory arbitrarily.

Example 1: Arithmetic chain `((a - b) * b) / scale`

```{code-block} python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
b_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
scale_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
tmp_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
out_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
tle.dsa.copy(b_ptrs, b_ub, [BM, BK])
tle.dsa.copy(scale_ptrs, scale_ub, [BM, BK])

tle.dsa.sub(a_ub, b_ub, tmp_ub)        # tmp = a - b
tle.dsa.mul(tmp_ub, b_ub, tmp_ub)      # tmp = tmp * b
tle.dsa.div(tmp_ub, scale_ub, out_ub)  # out = tmp / scale

tle.dsa.copy(out_ub, out_ptrs, [BM, BK])
```

Example 2: Clamp using `max` + `min`

```{code-block} python
x_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
floor_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
ceil_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
tmp_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)
y_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.UB)

tle.dsa.copy(x_ptrs, x_ub, [BM, BK])
tle.dsa.copy(floor_ptrs, floor_ub, [BM, BK])
tle.dsa.copy(ceil_ptrs, ceil_ub, [BM, BK])

tle.dsa.max(x_ub, floor_ub, tmp_ub)    # tmp = max(x, floor)
tle.dsa.min(tmp_ub, ceil_ub, y_ub)     # y = min(tmp, ceil)

tle.dsa.copy(y_ub, y_ptrs, [BM, BK])
```

## Loops and Hints

### tle.dsa.pipeline，tle.dsa.parallel，and tle.dsa.hint

Loops and Hints API include:

- `tle.dsa.pipeline(...)`
- `tle.dsa.parallel(...)`
- `tle.dsa.hint(...)` — provides compile-time hints in the form of a context manager `with tle.dsa.hint(...)`.

```{code-block} python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(x_ptr + offs, a_ub, [tail_size], inter_no_alias=True)
```

## Slicing and view

### tle.dsa.extract_slice, tle.dsa.insert_slice, tle.dsa.extract_element, and tle.dsa.subview

Slicing and view API include:

- `tle.dsa.extract_slice`
- `tle.dsa.insert_slice`
- `tle.dsa.extract_element`
- `tle.dsa.subview`

```{code-block} python
sub = tle.dsa.extract_slice(full, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
full = tle.dsa.insert_slice(full, sub, offsets=(0, k0), sizes=(BM, BK), strides=(1, 1))
elem = tle.dsa.extract_element(sub, indice=(i, j))
```
