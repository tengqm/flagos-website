# Hints

Hints provides a non-invasive performance hints injection mechanism that enables hardware-aware optimizations while maintaining full compatibility with native Triton code. The mechanism is simple: programmers add inline comments (`#@hint: <hint_name>`) to the corresponding Triton operations (for example, `tl.load`) to provide hardware-aware optimization hints. These hints are encoded as MLIR (Multi-Level Intermediate Representation) attributes during compilation, enabling the mid-end and backend to apply hardware-aware optimizations and multi-platform dynamic adaptation based on an elastic verification strategy.

This mechanism provides the following characteristics:

- Native compatibility: Hints are optional—kernels remain valid Triton and run correctly with the original Triton compiler.

- Low learning overhead: Hints are added via lightweight comments (`flagtree_hints`) without changing core Triton syntax.

- Enhanced compiler extensibility: New optimizations can be introduced by evolving hint schemas and MLIR attributes, avoiding language-level operation/syntax extensions.

- Enhanced performance capability: Hardware-aware hints unlock additional compiler optimizations to better utilize hardware features.

For Hints usage information, see [Use Hints](/user_guide/use-hints.md).
