# Tiered storage foundation

The foundation combines bounded io_uring completion windows, page-coalesced
plain-SSD slab operations, explicit DRAM admission budgets, profiled placement,
and persistent SGD updates for mixed DRAM/SSD batches. It does not enable online
admission by default.

`value.tiering.dram_keys_path` accepts a sorted, unique binary array of native
`uint64_t` keys. `high_watermark_ratio` bounds live DRAM reservations against the
configured allocator capacity. SSD residency statistics describe live rows;
allocator capacity, page-cache residency, and physical bytes are separate metrics.

Batch reads deduplicate pages and batch writes serialize overlapping page patches.
Fresh single-row allocations and batch allocations release their destinations
when an SSD write fails. A malformed row-length header cannot make a scalar read
cross its slot boundary. These guarantees do not make arbitrary multi-row writes
transactional after a device error.

Mixed-tier SGD aggregates gradients for duplicate keys, updates each existing row
once, initializes missing rows to zero, and persists SSD rows through the value
store rather than modifying temporary read buffers.

Run `test_io_backend`, `test_ssd_allocators`,
`test_ssd_static_layout_value_store`, `test_optimizer`, and `test_kvengine` from the
configured build's `bin/` directory. The allocator and value-store suites inject
SSD write failures, verify accounting recovery, and check slot-boundary reads.
