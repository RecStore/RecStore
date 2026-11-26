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
DEFINE_int32(rdma_cq_harvest_interval_ms,
             50,
             "Client lane completion harvest interval in milliseconds: a "
             "request waiting longer than this drains its lane CQ, so a QP "
             "completion error (retry exhaustion / flush) is classified and "
             "fails fast instead of waiting for the wait-timeout budget; 0 "
             "disables harvesting");
DEFINE_bool(
    rdma_lane_rebuild_enabled,
    false,
    "On a peer-side lane failure, rebuild the lane QP and resume service "
    "instead of retiring the lane for good. Requires both ends (server and "
    "client) to run with it enabled; off keeps the previous terminal-dirty "
    "behavior");
DEFINE_int32(rdma_lane_rebuild_max_attempts,
             3,
             "Rebuilds attempted per lane before the lane becomes Dead");
DEFINE_int32(rdma_lane_rebuild_wait_ms,
             2000,
             "Deadline for the bilateral rebuild handshake (a peer ack) in "
             "milliseconds");
DEFINE_int32(rdma_lane_rebuild_poll_ms,
             50,
             "Server-side interval between client lane-generation checks in "
             "milliseconds");
DEFINE_int32(
    rdma_lane_rebuild_poll_max_ms,
    500,
    "Upper bound for the idle backoff of the server-side lane-generation "
    "check. Each quiet check doubles the interval from "
    "--rdma_lane_rebuild_poll_ms up to this value; a rebuild resets it. Bounds "
    "the GetMeta load of an idle deployment while keeping the handshake inside "
    "--rdma_lane_rebuild_wait_ms");
DEFINE_int32(
    rdma_lane_rebuild_poll_timeout_ms,
    100,
    "Deadline for the GetMeta call issued by the server-side lane rebuild "
    "poll. The poll runs on a data-plane polling thread, so it must not "
    "inherit the 30s control-plane timeout");
DEFINE_int32(
    rdma_lane_rebuild_cooldown_ms,
    1000,
    "Minimum interval between rebuild attempts on the same lane, so a "
    "sustained outage does not burn the rebuild budget on every acquire in "
    "milliseconds");
DEFINE_int32(
    rdma_op_retry_max_attempts,
    2,
    "Retries for idempotent synchronous GET/PUT after a shard-unreachable "
    "error. Each retry acquires a new lane/slot; UPDATE never retries. "
    "0 disables retries");
DEFINE_int32(
    rdma_op_retry_backoff_ms,
    800,
    "Base backoff between idempotent GET/PUT retries; doubles per retry up "
    "to 2000 ms so a lane rebuild has time to finish");
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
DEFINE_int32(
    rdma_rc_server_get_worker_hold_us,
    0,
    "Test-only: hold each GET payload worker for this many microseconds "
    "before it handles the task, so a task provably straddles a lane rebuild "
    "and exercises the response-epoch fence. 0 disables the hold");
DEFINE_int32(
    rdma_rc_server_fence_stats_interval_ms,
    0,
    "Interval between rebuild-fence counter reports on the server stderr "
    "(`event=rebuild_fence_stats`). 0 disables the reports, which is the "
    "production default; test drivers enable them to assert the fence fired");
DEFINE_int32(
    rdma_rc_server_ready_delay_ms,
    0,
    "Test-only: delay the server's READY publication by this many "
    "milliseconds after the polling threads start. Lets a client reach "
    "WaitServer with the peer QPs already connected, then time out and retry. "
    "0 keeps the production behavior of publishing immediately");
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
DEFINE_int32(rdma_heartbeat_interval_ms,
             1000,
             "RDMA shard heartbeat interval in milliseconds");
DEFINE_int32(rdma_health_stale_ms,
             3000,
             "RDMA control-plane health stale threshold in milliseconds; a "
             "shard with no heartbeat within this window is Suspect");
DEFINE_int32(
    rdma_health_dead_ms,
    9000,
    "RDMA control-plane health dead threshold in milliseconds; a shard "
    "with no heartbeat within this window is Dead");
DEFINE_int32(rdma_health_probe_after_ms,
             1000,
             "Client slow-path health probe delay in milliseconds: a request "
             "waiting longer than this queries shard health from the shard-0 "
             "control plane; 0 disables slow-path probing");
DEFINE_int32(rdma_health_probe_interval_ms,
             1000,
             "Minimum interval between client slow-path health probes in "
             "milliseconds; also used as the TTL of the last verdict so "
             "concurrent waiters share one query");
DEFINE_int32(rdma_health_probe_timeout_ms,
             1000,
             "Deadline for a client slow-path health probe RPC in "
             "milliseconds; a probe that times out keeps the request on its "
             "data-path budgets");
DEFINE_string(rdma_rc_namespace,
              "",
              "Override RC write shared-memory namespace");
DEFINE_string(rdma_control_plane_host,
              "127.0.0.1",
              "Shard-0 RDMA control-plane TCP host");
DEFINE_string(rdma_rc_fake_get_mode,
              "none",
              "Benchmark-only fake GET mode: none, status_only, index_only, "
              "payload_memset, or throw (test-only handler fault injection)");
DEFINE_bool(rdma_rc_skip_client_copy,
            false,
            "Benchmark-only option to skip copying GET response payload from "
            "the RC response slot to the user receive buffer");
