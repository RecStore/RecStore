#include "ps/rdma/control_plane.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace petps {
namespace {

int AllocateTcpPort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  struct sockaddr_in addr {};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port        = 0;
  EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  socklen_t addr_len = sizeof(addr);
  EXPECT_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  const int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

TEST(RdmaControlPlaneTest, PublishMetaAndWaitServerReady) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      2000,
      "unit-test",
      1,
      1,
      "config",
      "fabric",
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  RawVerbsNodeMeta published{};
  published.node_id              = 1;
  published.deployment_id        = "unit-test";
  published.deployment_epoch     = 1;
  published.configuration_digest = "config";
  published.fabric_digest        = "fabric";
  published.gid[0]               = 1;
  published.port_num             = 1;
  published.gid_index            = 0;
  published.active_mtu           = IBV_MTU_1024;
  published.link_layer           = IBV_LINK_LAYER_INFINIBAND;
  published.base_addr            = 0x12345000ULL;
  published.rkey                 = 99;

  std::thread publisher([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    client.PublishMeta(1, 0, 2, 0, published);
    client.PublishServerReady(0);
    client.PublishServerReady(1);
  });

  const RawVerbsNodeMeta fetched = client.GetMeta(1, 0, 2, 0, 1000);
  EXPECT_EQ(fetched.node_id, published.node_id);
  EXPECT_EQ(fetched.base_addr, published.base_addr);
  EXPECT_EQ(fetched.rkey, published.rkey);

  EXPECT_NO_THROW(client.WaitServerReady(2, 1000));
  EXPECT_NO_THROW(client.WaitServer(1, 1000));

  publisher.join();
  server.Stop();
}

TEST(RdmaControlPlaneTest, VersionedMetadataRejectsInvalidWireData) {
  EXPECT_THROW(DecodeRawVerbsNodeMeta("not-protobuf"), std::runtime_error);
  EXPECT_THROW(
      DecodeRawVerbsNodeMeta(std::string(kMaxRawVerbsMetadataBytes + 1, '\0')),
      std::runtime_error);

  RawVerbsNodeMeta meta{};
  meta.deployment_id        = "unit-test";
  meta.deployment_epoch     = 1;
  meta.configuration_digest = "config";
  meta.fabric_digest        = "fabric";
  meta.protocol_version     = 2;
  meta.port_num             = 1;
  meta.gid_index            = 0;
  meta.active_mtu           = IBV_MTU_1024;
  EXPECT_THROW(DecodeRawVerbsNodeMeta(EncodeRawVerbsNodeMeta(meta)),
               std::runtime_error);

  meta.protocol_version = 1;
  meta.link_layer       = 99;
  EXPECT_THROW(DecodeRawVerbsNodeMeta(EncodeRawVerbsNodeMeta(meta)),
               std::runtime_error);
}

TEST(RdmaControlPlaneTest, RejectsMetadataFromAnotherDeployment) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      2000,
      "unit-test",
      2,
      1,
      "config",
      "fabric"};
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneEndpoint stale_endpoint = endpoint;
  stale_endpoint.deployment_id            = "old-deployment";
  RdmaControlPlaneClient stale_client(stale_endpoint);
  RawVerbsNodeMeta meta{};
  meta.node_id              = 0;
  meta.deployment_id        = stale_endpoint.deployment_id;
  meta.deployment_epoch     = stale_endpoint.deployment_epoch;
  meta.configuration_digest = "config";
  meta.fabric_digest        = "fabric";
  meta.port_num             = 1;
  meta.gid_index            = 0;
  meta.active_mtu           = IBV_MTU_1024;
  meta.link_layer           = IBV_LINK_LAYER_INFINIBAND;
  EXPECT_THROW(stale_client.PublishMeta(0, 0, 1, 0, meta), std::runtime_error);
  EXPECT_THROW(stale_client.GetMeta(0, 0, 1, 0, 50), std::runtime_error);
  EXPECT_THROW(stale_client.PublishServerReady(0), std::runtime_error);

  server.Stop();
}

TEST(RdmaControlPlaneTest, PeerValidationRejectsStaleEpoch) {
  RawVerbsConfig local;
  local.num_servers          = 1;
  local.num_clients          = 1;
  local.deployment_id        = "unit-test";
  local.deployment_epoch     = 2;
  local.configuration_digest = "config";
  local.fabric_digest        = "fabric";
  local.port_num             = 1;
  local.fabric_mode          = recstore::RdmaFabricMode::kIb;

  RawVerbsNodeMeta remote{};
  remote.node_id    = 0;
  remote.logical_id = 0;
  remote.node_role = static_cast<std::uint8_t>(recstore::RdmaNodeRole::kServer);
  remote.deployment_id        = "unit-test";
  remote.deployment_epoch     = 1;
  remote.configuration_digest = "config";
  remote.fabric_digest        = "fabric";
  remote.port_num             = 2;
  remote.gid_index            = 0;
  remote.link_layer           = IBV_LINK_LAYER_INFINIBAND;
  remote.active_mtu           = IBV_MTU_1024;
  remote.fabric_mode = static_cast<std::uint8_t>(recstore::RdmaFabricMode::kIb);

  EXPECT_THROW(ValidateRawVerbsPeerMeta(local, 0, remote), std::runtime_error);
  remote.deployment_epoch = 2;
  EXPECT_NO_THROW(ValidateRawVerbsPeerMeta(local, 0, remote));
  remote.port_num = 0;
  EXPECT_THROW(ValidateRawVerbsPeerMeta(local, 0, remote), std::runtime_error);
  remote.port_num  = 2;
  remote.gid_index = -1;
  EXPECT_THROW(ValidateRawVerbsPeerMeta(local, 0, remote), std::runtime_error);
}

TEST(RdmaControlPlaneTest, ReadyRejectsMismatchedConfigurationDigest) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient publisher(endpoint);
  publisher.PublishServerReady(0);

  endpoint.configuration_digest = "other-config";
  RdmaControlPlaneClient waiter(endpoint);
  EXPECT_THROW(waiter.WaitServer(0, 50), std::runtime_error);
  EXPECT_THROW(waiter.WaitServerReady(1, 50), std::runtime_error);
  server.Stop();
}

TEST(RdmaControlPlaneTest, WaitSpecificServerTimesOut) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
      "unit-test",
      1,
      1,
      "config",
      "fabric",
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  EXPECT_THROW(client.WaitServer(3, 50), std::runtime_error);

  server.Stop();
}

TEST(RdmaControlPlaneTest, WaitServerReadyTimesOut) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
      "unit-test",
      1,
      1,
      "config",
      "fabric",
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  EXPECT_THROW(client.WaitServerReady(1, 50), std::runtime_error);

  server.Stop();
}

TEST(RdmaControlPlaneTest, DrainingServerIsNoLongerReady) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
      "unit-test",
      1,
      1,
      "config",
      "fabric",
      1,
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  client.PublishServerReady(0);
  EXPECT_NO_THROW(client.WaitServer(0, 50));
  client.PublishServerDraining(0);
  EXPECT_THROW(client.WaitServer(0, 50), std::runtime_error);
  EXPECT_THROW(client.WaitServerReady(1, 50), std::runtime_error);
  EXPECT_THROW(client.PublishServerReady(0), std::runtime_error);

  server.Stop();
}

TEST(RdmaControlPlaneTest, ServerMustBeReadyBeforeDraining) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
      "unit-test",
      1,
      1,
      "config",
      "fabric",
      1,
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  EXPECT_THROW(client.PublishServerDraining(0), std::runtime_error);

  server.Stop();
}

TEST(RdmaControlPlaneTest, EvaluateServerHealthBoundaries) {
  using std::chrono::steady_clock;
  const auto base = steady_clock::time_point{} + std::chrono::seconds(10);
  const auto at = [&](int ms) { return base + std::chrono::milliseconds(ms); };

  // A record that was never touched is never reported dead.
  EXPECT_EQ(EvaluateServerHealth(steady_clock::time_point{}, at(0), 3000, 9000),
            RdmaServerHealth::kServing);
  EXPECT_EQ(EvaluateServerHealth(base, at(10), 3000, 9000),
            RdmaServerHealth::kServing);
  EXPECT_EQ(EvaluateServerHealth(base, at(2999), 3000, 9000),
            RdmaServerHealth::kServing);
  EXPECT_EQ(EvaluateServerHealth(base, at(3000), 3000, 9000),
            RdmaServerHealth::kSuspect);
  EXPECT_EQ(EvaluateServerHealth(base, at(8999), 3000, 9000),
            RdmaServerHealth::kSuspect);
  EXPECT_EQ(EvaluateServerHealth(base, at(9000), 3000, 9000),
            RdmaServerHealth::kDead);
}

TEST(RdmaControlPlaneTest, HeartbeatRefreshesLivenessAndTracksInstance) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  endpoint.health_stale_ms            = 300;
  endpoint.health_dead_ms             = 900;
  endpoint.health_record_retention_ms = 60000;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0);
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kServing);

  // No heartbeat inside the stale window -> suspect, then dead.
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kSuspect);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kDead);

  // A heartbeat brings it back to serving and records the instance nonce.
  client.SendHeartbeat(0, "instance-a");
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kServing);
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-a");
  EXPECT_GE(server.heartbeats_received(), 1u);

  client.SendHeartbeat(0, "instance-b");
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-b");

  server.Stop();
}

TEST(RdmaControlPlaneTest, ReadyCarriesInstanceNonce) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0, "instance-a");
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-a");

  // A READY without a nonce must not wipe the recorded identity.
  client.PublishServerReady(0);
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-a");

  // A READY from a new process replaces it.
  client.PublishServerReady(0, "instance-b");
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-b");

  server.Stop();
}

TEST(RdmaControlPlaneTest, HealthRecordEvictedAfterRetention) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  endpoint.health_stale_ms            = 60;
  endpoint.health_dead_ms             = 120;
  endpoint.health_record_retention_ms = 150;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0);
  client.SendHeartbeat(0, "instance-a");
  const std::uint64_t recorded = server.heartbeats_received();
  EXPECT_GE(recorded, 1u);

  // Past the retention window the monitor evicts the record; heartbeats for an
  // unknown shard are ignored (READY is the only way to become serving).
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  client.SendHeartbeat(0, "instance-a");
  EXPECT_EQ(server.heartbeats_received(), recorded);
  EXPECT_TRUE(server.ServerInstanceNonce(0).empty());
  server.Stop();
}

TEST(RdmaControlPlaneTest, HealthRecordRetainedWhenGcDisabled) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  endpoint.health_stale_ms            = 30;
  endpoint.health_dead_ms             = 60;
  endpoint.health_record_retention_ms = 0; // GC disabled (default)
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0, "instance-a");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // The record survives its own death when eviction is disabled; heartbeats for
  // it are still accepted (which is how it recovers without a new READY).
  client.SendHeartbeat(0, "instance-a");
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-a");
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kServing);

  server.Stop();
}

TEST(RdmaControlPlaneTest, GetServerHealthReportsShardState) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  endpoint.health_stale_ms = 30;
  endpoint.health_dead_ms  = 60;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0, "instance-a");

  bool query_ok                     = false;
  RdmaServerHealthSnapshot snapshot = client.GetServerHealth(0, -1, &query_ok);
  EXPECT_TRUE(query_ok);
  EXPECT_TRUE(snapshot.known);
  EXPECT_EQ(snapshot.health, RdmaServerHealth::kServing);
  EXPECT_GE(snapshot.last_seen_age_ms, 0);
  EXPECT_EQ(snapshot.instance_nonce, "instance-a");

  // Unknown shard: a successful query that carries no opinion.
  query_ok = false;
  snapshot = client.GetServerHealth(7, -1, &query_ok);
  EXPECT_TRUE(query_ok);
  EXPECT_FALSE(snapshot.known);

  // No heartbeat: health is evaluated fresh at query time.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  query_ok = false;
  snapshot = client.GetServerHealth(0, -1, &query_ok);
  EXPECT_TRUE(query_ok);
  EXPECT_TRUE(snapshot.known);
  EXPECT_EQ(snapshot.health, RdmaServerHealth::kDead);
  EXPECT_GE(snapshot.last_seen_age_ms, 300);

  // A heartbeat brings the same instance back to serving.
  client.SendHeartbeat(0, "instance-a");
  snapshot = client.GetServerHealth(0);
  EXPECT_TRUE(snapshot.known);
  EXPECT_EQ(snapshot.health, RdmaServerHealth::kServing);

  server.Stop();
}

TEST(RdmaControlPlaneTest, GetServerHealthTreatsErrorsAsUnknown) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);
  client.PublishServerReady(0, "instance-a");

  // Wrong deployment identity is rejected by the server; the client reports an
  // unknown snapshot instead of leaking a health opinion.
  RdmaControlPlaneEndpoint other = endpoint;
  other.deployment_id            = "other-deployment";
  RdmaControlPlaneClient foreign_client(other);
  bool query_ok = true;
  RdmaServerHealthSnapshot snapshot =
      foreign_client.GetServerHealth(0, -1, &query_ok);
  EXPECT_FALSE(query_ok);
  EXPECT_FALSE(snapshot.known);

  // Control plane down: still no opinion, no throw.
  server.Stop();
  query_ok = true;
  snapshot = client.GetServerHealth(0, -1, &query_ok);
  EXPECT_FALSE(query_ok);
  EXPECT_FALSE(snapshot.known);
}

TEST(RdmaControlPlaneTest, SlowPathVerdictOnlyFailsOnDead) {
  RdmaServerHealthSnapshot dead;
  dead.known  = true;
  dead.health = RdmaServerHealth::kDead;
  EXPECT_EQ(ClassifyShardHealthSnapshot(true, dead),
            RdmaSlowPathVerdict::kShardUnreachable);

  RdmaServerHealthSnapshot suspect;
  suspect.known  = true;
  suspect.health = RdmaServerHealth::kSuspect;
  EXPECT_EQ(ClassifyShardHealthSnapshot(true, suspect),
            RdmaSlowPathVerdict::kKeepWaiting);

  RdmaServerHealthSnapshot serving;
  serving.known  = true;
  serving.health = RdmaServerHealth::kServing;
  EXPECT_EQ(ClassifyShardHealthSnapshot(true, serving),
            RdmaSlowPathVerdict::kKeepWaiting);

  // Unknown shard: the control plane has no READY record for it.
  RdmaServerHealthSnapshot unknown;
  EXPECT_EQ(ClassifyShardHealthSnapshot(true, unknown),
            RdmaSlowPathVerdict::kKeepWaiting);

  // Failed query (control plane unreachable): never a false verdict, even if a
  // stale snapshot claims death.
  EXPECT_EQ(ClassifyShardHealthSnapshot(false, dead),
            RdmaSlowPathVerdict::kKeepWaiting);
}

TEST(RdmaControlPlaneTest, SlowPathProbeGateThrottlesAndCoalesces) {
  RdmaSlowPathProbeGate gate;
  const auto start = std::chrono::steady_clock::now();

  // First probe is admitted.
  EXPECT_TRUE(gate.TryBeginProbe(start, 100));
  // A concurrent waiter is coalesced while the probe is in flight.
  EXPECT_FALSE(gate.TryBeginProbe(start, 100));
  gate.EndProbe();
  EXPECT_EQ(gate.probes(), 1u);
  EXPECT_EQ(gate.skipped(), 1u);

  // Within the TTL the verdict is considered fresh.
  EXPECT_FALSE(gate.TryBeginProbe(start + std::chrono::milliseconds(99), 100));
  // A ttl of 0 disables the freshness window but still coalesces in-flight
  // probes.
  EXPECT_TRUE(gate.TryBeginProbe(start + std::chrono::milliseconds(100), 0));
  EXPECT_FALSE(gate.TryBeginProbe(start + std::chrono::milliseconds(100), 0));
  gate.EndProbe();

  // After the TTL a new probe runs again.
  EXPECT_TRUE(gate.TryBeginProbe(start + std::chrono::milliseconds(200), 100));
  gate.EndProbe();
  EXPECT_EQ(gate.probes(), 3u);
}

TEST(RdmaControlPlaneTest, HeartbeatIdentityIsValidated) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  endpoint.num_servers     = 1;
  endpoint.health_stale_ms = 60;
  endpoint.health_dead_ms  = 120;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0, "instance-a");
  client.SendHeartbeat(0, "instance-a");
  ASSERT_EQ(server.heartbeats_received(), 1u);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  ASSERT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kDead);

  // Another deployment cannot revive this shard's record.
  RdmaControlPlaneEndpoint foreign = endpoint;
  foreign.deployment_id            = "other-deployment";
  RdmaControlPlaneClient foreign_client(foreign);
  foreign_client.SendHeartbeat(0, "instance-a");
  EXPECT_EQ(server.heartbeats_rejected(), 1u);
  EXPECT_EQ(server.heartbeats_received(), 1u);
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kDead);

  // A new epoch of the same deployment is rejected as well.
  RdmaControlPlaneEndpoint next_epoch = endpoint;
  next_epoch.deployment_epoch         = 2;
  RdmaControlPlaneClient epoch_client(next_epoch);
  epoch_client.SendHeartbeat(0, "instance-a");
  EXPECT_EQ(server.heartbeats_rejected(), 2u);
  EXPECT_EQ(server.heartbeats_received(), 1u);

  // A shard id outside num_servers is rejected before the record lookup.
  client.SendHeartbeat(1, "instance-a");
  EXPECT_EQ(server.heartbeats_rejected(), 3u);
  EXPECT_EQ(server.heartbeats_unknown_shard(), 0u);

  // The shard's own heartbeat still refreshes the record.
  client.SendHeartbeat(0, "instance-b");
  EXPECT_EQ(server.heartbeats_received(), 2u);
  EXPECT_EQ(server.ServerHealth(0, std::chrono::steady_clock::now()),
            RdmaServerHealth::kServing);
  EXPECT_EQ(server.ServerInstanceNonce(0), "instance-b");

  server.Stop();
}

TEST(RdmaControlPlaneTest, HeartbeatForUnknownShardIsCounted) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  endpoint.num_servers = 2;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  // No READY record yet: a heartbeat never creates one.
  client.SendHeartbeat(1, "instance-a");
  EXPECT_EQ(server.heartbeats_unknown_shard(), 1u);
  EXPECT_EQ(server.heartbeats_received(), 0u);
  EXPECT_FALSE(server.ServerHealthSnapshot(1).known);

  // After READY the same heartbeat is accepted.
  client.PublishServerReady(1, "instance-a");
  client.SendHeartbeat(1, "instance-a");
  EXPECT_EQ(server.heartbeats_unknown_shard(), 1u);
  EXPECT_EQ(server.heartbeats_received(), 1u);

  server.Stop();
}

TEST(RdmaControlPlaneTest, ControlPlaneClientServesManyRpcCalls) {
  RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      500,
      "unit-test",
      1,
      1,
      "config",
      "fabric"};
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0, "instance-a");
  client.SendHeartbeat(0, "instance-a");
  bool query_ok = false;
  const RdmaServerHealthSnapshot snapshot =
      client.GetServerHealth(0, -1, &query_ok);
  EXPECT_TRUE(query_ok);
  EXPECT_TRUE(snapshot.known);
  client.WaitServer(0, 500);
  client.SendHeartbeat(0, "instance-a");

  // Different RPC kinds on one lazily opened, shared channel.
  EXPECT_EQ(server.heartbeats_received(), 2u);

  server.Stop();
}

} // namespace
} // namespace petps
