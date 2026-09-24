#include "ps/rdma/rc_options.h"

DEFINE_int32(rdma_rc_qps_per_client_per_shard,
             16,
             "RC write QPs per client per shard");
DEFINE_int32(rdma_rc_slots_per_qp,
             1,
             "Logical request/response slots per RC write QP");
DEFINE_int32(rdma_rc_mtu_bytes, 4096, "RC write logical MTU bytes");
DEFINE_int32(rdma_rc_target_response_mtu,
             200,
             "RC write target GET response MTU count");
DEFINE_int32(rdma_rc_request_slot_bytes,
             1 << 20,
             "RC write request slot bytes");
DEFINE_int32(rdma_rc_response_slot_bytes,
             1 << 20,
             "RC write response slot bytes");
DEFINE_int32(rdma_wait_timeout_ms,
             60000,
             "RC write single RPC wait timeout in milliseconds");
DEFINE_int32(rdma_rc_profile_interval_ms,
             0,
             "RC write profiling summary interval in milliseconds; 0 disables "
             "profiling");
DEFINE_int32(rdma_rc_server_coroutines_per_thread,
             1,
             "RC write server coroutine count per poll thread. Values greater "
             "than 1 enable cooperative slot scanning inside each poll thread");
DEFINE_int32(rdma_rc_server_get_workers,
             0,
             "Experimental RC write server GET payload worker thread count. "
             "0 keeps GET handling on the polling thread");
DEFINE_int32(rdma_rc_client_id_base,
             -1,
             "Logical RC write client id base for this OS process. A negative "
             "value derives the id from global_id for backward compatibility");
DEFINE_int32(rdma_rc_num_logical_clients,
             -1,
             "Total logical RC write clients used by the slot protocol. A "
             "negative value uses num_client_processes for compatibility");
DEFINE_int32(rdma_rc_wait_spin_iterations,
             0,
             "Client-side status polling spin iterations before yielding. "
             "Higher values reduce scheduler handoff overhead for low-latency "
             "RDMA completions at the cost of more CPU busy polling");
DEFINE_int32(rdma_rc_inline_bytes,
             64,
             "Requested RC write inline-data threshold in bytes. Small RDMA "
             "writes at or below the granted device limit use IBV_SEND_INLINE");
DEFINE_int32(rdma_rc_client_numa_id,
             0,
             "RDMA device index/NUMA hint used by RC benchmark and client-side "
             "transport");
DEFINE_int32(rdma_rc_server_numa_id,
             0,
             "RDMA device index/NUMA hint used by RC server-side transport");
DEFINE_int32(rdma_control_plane_port,
             25100,
             "Shard-0 RDMA control-plane TCP port");
DEFINE_int32(rdma_control_plane_timeout_ms,
             30000,
             "RDMA control-plane request timeout in milliseconds");
DEFINE_string(rdma_rc_namespace,
              "",
              "Override RC write shared-memory namespace");
DEFINE_string(rdma_control_plane_host,
              "127.0.0.1",
              "Shard-0 RDMA control-plane TCP host");
DEFINE_string(rdma_rc_fake_get_mode,
              "none",
              "Benchmark-only fake GET mode: none, status_only, index_only, "
              "or payload_memset");
DEFINE_bool(rdma_rc_skip_client_copy,
            false,
            "Benchmark-only option to skip copying GET response payload from "
            "the RC response slot to the user receive buffer");
