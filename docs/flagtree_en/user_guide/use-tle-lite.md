# Use TLE-Lite

This section introduces how to use TLE-Lite. TLE-Lite is available on trition_3.6.x branch.

## Memory management

You can use the following operations to manage the memory.

### tle.load

`tle.load` loads a tensor asynchronously from GMEM. It supports asynchronously hint.

```{code-block} python
x = tle.load(..., is_async=True)
```

## Tensor slicing

Splits the input tensor into a grid of sub-tiles based on the specified sub-tile shape, and extracts the sub-tile at the given coordinates.
GPU: Supports extraction into registers and shared memory.

### tle.extract_tile

Splits the input tensor into a grid of sub-tiles based on the specified sub-tile shape, and extracts the sub-tile at the given coordinates.

Supports extraction into registers and shared memory.

```{code-block} python
# x is [4, 4]
# z is [2, 2]
# Split x into a sub-tile grid with shape=[2, 2], and extract the sub-tile at [0, 0]
z = x.extract_tile(index=[0, 0], shape=[2, 2])
```

### tle.insert_tile

Splits the input tensor into a grid of sub-tiles based on the sub-tile shape, and updates the sub-tile at the specified coordinates with a new tile.

Supports updates from registers and shared memory.

```{code-block} python
# x is [4, 4], y is [2, 2], z is [4, 4]
# Split x into sub-tiles of shape=[2, 2], update the [0, 0] sub-tile with y, and return the full [4, 4] tensor
z = x.insert_tile(y, index=[0, 0])
```

## Distribution

The Triton distributed API consists of four core parts: device mesh definition, sharding specification description, synchronization, and remote access (point-to-point communication).

### device mesh

#### tle.device_mesh

`tle.device_mesh` defines the topological structure of physical devices. It is the fundamental context for all distributed operations.

```{code-block} python
class device_mesh:
    def __init__(self, topology: dict):
        """
        Initialize a DeviceMesh.

        Args:
            topology (dict): A dictionary describing the hardware hierarchy.
                             Keys are level names; values are either an integer (for 1D)
                             or a list of tuples (for multi-dimensional levels).
        """
        self._physical_ids = ...  # Internal storage: flattened list of physical IDs (0..N-1)
        self._shape = ...         # Shape of the current logical view, e.g., (2, 2, 4, 2, 2, 4)
        self._dim_names = ...     # Names of the current dimensions
        # Initialization and parsing logic...

    @property
    def shape(self):
        """Return the logical shape of the current mesh."""
        return self._shape

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return len(self._shape)

    def flatten(self):
        """
        Flatten the mesh into 1D. Commonly used for ring-based communication patterns.
        """
        return self.reshape(prod(self._shape))

    def __getitem__(self, key):
        """
        Support slicing operations and return a sub-mesh.
        Supports standard slices (slice objects) and integer indexing.
        """
        # Compute new shape and selected physical IDs after slicing
        # ...
        return sub_mesh

    def __repr__(self):
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


# Define a complex hardware hierarchy
topology = {
    # Inter-node level (2x2 = 4 nodes)
    "node": [("node_x", 2), ("node_y", 2)],
    # Intra-node GPUs (4 devices)
    "device": 4,
    # Intra-GPU clusters (2x2)
    "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
    # Blocks within each cluster (4 blocks)
    "block": 4
}

# mesh.shape -> (2, 2, 4, 2, 2, 4)
# Total size = 256
mesh = tle.device_mesh(topology=topology)
```

### Sharding specification

`tle.sharding` is used to declare the current distribution state of a tensor across a Device Mesh. The splits list describes how each dimension of the tensor is partitioned over the mesh, while the partials list indicates whether the tensor is in a partial-sum state. Any mesh axes not explicitly mentioned are treated as broadcast (replicated).

- tle.S(axis): Split — indicates that the tensor dimension is partitioned along the specified mesh axis.
- tle.B: Broadcast/Replicate — indicates that the tensor dimension is fully replicated (i.e., not split) along any mesh axes not explicitly referenced.
- tle.P(axis): Partial — indicates that the tensor holds only a partial value (e.g., a partial sum) and must be reduced along the specified mesh axis to obtain the complete result.

```{code-block} python
def sharding(tensor, splits, partials):
    """
    Annotation: Used only to annotate the tensor's layout state.
    It does not generate any runtime code but guides the compiler for subsequent optimizations or correctness checks.
    """
    return tensor


# Define a sharding spec where:
# - axis 0 is split across the "cluster" dimension (specifically over ["cluster_x", "cluster_y"]),
# - axis 1 is split across the "device" dimension,
# - and the tensor is in a partial state along the "block" dimension (requiring a reduce to resolve).
x_shard = tle.sharding(
    mesh,
    split=[["cluster_x", "cluster_y"], "device"],
    partial=["block"]
)

# Create a sharded tensor using the above sharding specification
x = tle.make_sharded_tensor(x_ptr, sharding=x_shard, shape=[4, 4])
```

### Synchronization

In complex distributed operators—such as Ring-AllReduce or pipelined execution with independent row/column communication—we often need to synchronize only thread blocks within the same "row" or "column," rather than across the entire cluster. A global synchronization would introduce unnecessary waiting overhead.
This API supports sub-mesh synchronization, meaning that within a large physical cluster, we can define multiple logical "communication groups" and perform synchronization independently within each group.

```{code-block} python
def distributed_barrier(mesh):
    """
    If a sub-mesh is passed, only devices within that sub-mesh are synchronized.
    Devices outside the sub-mesh should treat this instruction as a no-op 
    (or the compiler should ensure their control flow never reaches this point).
    """
    pass
```

#### tle.distributed_barrier

`tle.distributed_barrier` synchronize only the set of devices corresponding to the given mesh or sub-mesh.

Read from neighboring shards (ring-style exchange).

```{code-block} python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

### Remote access

`tle.remote` is used to obtain a handle to a tensor located on another device. This corresponds to point-to-point communication or direct memory access (e.g., RDMA/NVLink Load). It enables kernels to explicitly access data from a specific shard.

```{code-block} python
def remote(tensor, shard_id, scope):
    """
    Obtains a handle to a Remote Tensor residing on a specific device shard.

    :param tensor: A logically distributed tensor (already annotated with tle.sharding).
    :param shard_id: tuple. The coordinates of the target device within the Device Mesh.
                     For example, if mesh=(2,4) and shard_id=(0, 3), this refers to GPU #3 on node #0.
    :return: RemoteTensor. Supports operations such as load, store, etc.
    """
```

`tle.remote`: Explicitly read from or write to remote shards.

```{code-block} python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

## Primitives interactive with local_ptr

The following APIs are used together with `tle.gpu.local_ptr`. For more information, see [Use TLE-Struct](use-tle-struct.md).

- `tl.load`（for local_ptr）
- `tl.store`（for local_ptr）
- `tl.atomic_add`/`and`/`cas`/`max`/`min`/`or`/`xchg`/`xor`（for local_ptr）