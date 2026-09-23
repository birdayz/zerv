# GGUF parse/index component benchmark

Research prerequisite: [GGUF loading](../research/gguf-loading.md). The pinned
`gguf_init_from_buffer(const void *, size_t, gguf_init_params)` parses the supplied
buffer synchronously and owns metadata strings/indexes; `no_alloc=true, ctx=null`
prevents tensor body allocation. `gguf_free` destroys that context. The native
parser instead owns indexes and borrows all bytes. Both parse and destruction
belong inside the timing. This is an asymmetric startup-component comparison,
not equivalent memory ownership and not model loading onto GPU or serving.

The checked-in harness must:

- Build native ReleaseFast benchmark/inspector and run both native test modes.
- Accept exactly one immutable GGUF file, one explicit external ggml library and
  a new output directory. Reject a preexisting output directory.
- Record complete model SHA-256/size, reference binary/version/commit, Zig version,
  binary/source hashes, host/kernel, effective CPU affinity, raw times and commands.
- Before timing, run native and independent reference inspection and require exact
  equality of all fields (metadata hashes, tensor descriptors and payload samples).
  Hashing and inspection are outside timing; this is a warm-header/page-cache test.
- Map file once per worker. Python uses private copy-on-write mmap solely to expose
  a stable ctypes buffer pointer; neither worker writes input bytes. Native uses
  read-only private mapping. Neither touches all weight pages inside timing.
- Use 3 warmups, 7 trials of 10 parse/free cycles per worker. Run 3 alternating-order
  rounds pinned to the same explicitly reported allowed logical CPU. No concurrent
  workers. Native uses page_allocator; reference uses its library allocator. Do not
  conceal UTF-8 validation, payload-bounds checking, or metadata-copy differences.
- Return strict JSONL trial records containing elapsed_ns, trial, iterations,
  metadata/tensor counts and data_offset. Refuse missing/duplicate/extra/invalid
  trials, nonpositive times or descriptor count/offset mismatches.
- Report medians/ranges of per-parse time and ratio; retain every raw trial.

Unit tests cover result-validation failures. Correctness failure aborts measurement;
ordinary native tests/builds never load the external library. The harness is an
external benchmark tool, not a production inference dependency. A result does not
imply tensor numerical validation, kernel support, or successful Qwen generation.
