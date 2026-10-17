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

// The health tests share one endpoint contract and only vary the port plus a
// couple of thresholds.
RdmaControlPlaneEndpoint
MakeHealthEndpoint(int timeout_ms = 500, int num_servers = 0) {
  RdmaControlPlaneEndpoint endpoint;
  endpoint.port                 = AllocateTcpPort();
  endpoint.timeout_ms           = timeout_ms;
  endpoint.deployment_id        = "unit-test";
  endpoint.deployment_epoch     = 1;
  endpoint.protocol_version     = 1;
  endpoint.configuration_digest = "config";
  endpoint.fabric_digest        = "fabric";
  endpoint.num_servers          = num_servers;
  return endpoint;
}

RdmaServerHealth HealthOf(const RdmaControlPlaneServer& server, int server_id) {
  return server.ServerHealthSnapshot(server_id).health;
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

TEST(RdmaControlPlaneTest, HeartbeatRefreshesLivenessPastDeath) {
  RdmaControlPlaneEndpoint endpoint = MakeHealthEndpoint();
  endpoint.health_stale_ms          = 300;
  endpoint.health_dead_ms           = 900;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0);
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kServing);

  // No heartbeat inside the stale window -> suspect, then dead.
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kSuspect);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kDead);

  // A heartbeat revives the record without a new READY.
  client.SendHeartbeat(0);
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kServing);

  server.Stop();
}

TEST(RdmaControlPlaneTest, GetServerHealthReportsShardState) {
  RdmaControlPlaneEndpoint endpoint = MakeHealthEndpoint();
  endpoint.health_stale_ms          = 30;
  endpoint.health_dead_ms           = 60;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0);

  RdmaServerHealthSnapshot snapshot = client.GetServerHealth(0, -1);
  EXPECT_TRUE(snapshot.known);
  EXPECT_EQ(snapshot.health, RdmaServerHealth::kServing);
  EXPECT_GE(snapshot.last_seen_age_ms, 0);

  // Unknown shard: a successful query that carries no opinion.
  snapshot = client.GetServerHealth(7, -1);
  EXPECT_FALSE(snapshot.known);

  // No heartbeat: health is evaluated fresh at query time.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  snapshot = client.GetServerHealth(0, -1);
  EXPECT_TRUE(snapshot.known);
  EXPECT_EQ(snapshot.health, RdmaServerHealth::kDead);
  EXPECT_GE(snapshot.last_seen_age_ms, 300);

  // A heartbeat brings the same instance back to serving.
  client.SendHeartbeat(0);
  snapshot = client.GetServerHealth(0);
  EXPECT_TRUE(snapshot.known);
  EXPECT_EQ(snapshot.health, RdmaServerHealth::kServing);

  server.Stop();
}

TEST(RdmaControlPlaneTest, GetServerHealthTreatsErrorsAsUnknown) {
  RdmaControlPlaneEndpoint endpoint = MakeHealthEndpoint(200);
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);
  client.PublishServerReady(0);

  // Wrong deployment identity is rejected by the server; the client reports an
  // unknown snapshot instead of leaking a health opinion.
  RdmaControlPlaneEndpoint other = endpoint;
  other.deployment_id            = "other-deployment";
  RdmaControlPlaneClient foreign_client(other);
  RdmaServerHealthSnapshot snapshot = foreign_client.GetServerHealth(0, -1);
  EXPECT_FALSE(snapshot.known);

  // Control plane down: still no opinion, no throw.
  server.Stop();
  snapshot = client.GetServerHealth(0, -1);
  EXPECT_FALSE(snapshot.known);
}

TEST(RdmaControlPlaneTest, SlowPathOnlyFailsOnDeadShards) {
  RdmaServerHealthSnapshot dead;
  dead.known  = true;
  dead.health = RdmaServerHealth::kDead;
  EXPECT_TRUE(IsShardUnreachable(dead));

  RdmaServerHealthSnapshot suspect;
  suspect.known  = true;
  suspect.health = RdmaServerHealth::kSuspect;
  EXPECT_FALSE(IsShardUnreachable(suspect));

  RdmaServerHealthSnapshot serving;
  serving.known  = true;
  serving.health = RdmaServerHealth::kServing;
  EXPECT_FALSE(IsShardUnreachable(serving));

  // Unknown shard (or an unreachable control plane): the control plane carries
  // no opinion, so the request stays on its data-path budgets.
  const RdmaServerHealthSnapshot unknown;
  EXPECT_FALSE(IsShardUnreachable(unknown));
}

TEST(RdmaControlPlaneTest, SlowPathProbeGateThrottlesAndCoalesces) {
  RdmaSlowPathProbeGate gate;
  const auto start = std::chrono::steady_clock::now();

  // First probe is admitted; a concurrent waiter is coalesced while it runs.
  EXPECT_TRUE(gate.TryBeginProbe(start, 100));
  EXPECT_FALSE(gate.TryBeginProbe(start, 100));
  gate.EndProbe();

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
}

TEST(RdmaControlPlaneTest, HeartbeatIdentityIsValidated) {
  RdmaControlPlaneEndpoint endpoint =
      MakeHealthEndpoint(500, /*num_servers=*/1);
  endpoint.health_stale_ms = 60;
  endpoint.health_dead_ms  = 120;
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0);
  client.SendHeartbeat(0);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  ASSERT_EQ(HealthOf(server, 0), RdmaServerHealth::kDead);

  // Another deployment is rejected (the sender logs it: a heartbeat is
  // observational and never fatal) and cannot revive this shard's record.
  RdmaControlPlaneEndpoint foreign = endpoint;
  foreign.deployment_id            = "other-deployment";
  RdmaControlPlaneClient foreign_client(foreign);
  EXPECT_THROW(foreign_client.SendHeartbeat(0), std::runtime_error);
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kDead);

  // A new epoch of the same deployment is rejected as well.
  RdmaControlPlaneEndpoint next_epoch = endpoint;
  next_epoch.deployment_epoch         = 2;
  RdmaControlPlaneClient epoch_client(next_epoch);
  EXPECT_THROW(epoch_client.SendHeartbeat(0), std::runtime_error);
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kDead);

  // A shard id with no READY record refreshes nothing and creates nothing.
  client.SendHeartbeat(1);
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kDead);

  // The shard's own heartbeat still refreshes the record.
  client.SendHeartbeat(0);
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kServing);

  server.Stop();
}

TEST(RdmaControlPlaneTest, HeartbeatWithoutReadyRecordIsIgnored) {
  RdmaControlPlaneEndpoint endpoint =
      MakeHealthEndpoint(500, /*num_servers=*/2);
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  // No READY record yet: a heartbeat never creates one.
  client.SendHeartbeat(1);
  EXPECT_FALSE(server.ServerHealthSnapshot(1).known);

  // After READY the same heartbeat is accepted and refreshes `last_seen`.
  client.PublishServerReady(1);
  client.SendHeartbeat(1);
  EXPECT_TRUE(server.ServerHealthSnapshot(1).known);
  EXPECT_GE(server.ServerHealthSnapshot(1).last_seen_age_ms, 0);

  server.Stop();
}

TEST(RdmaControlPlaneTest, ControlPlaneClientServesManyRpcCalls) {
  RdmaControlPlaneEndpoint endpoint = MakeHealthEndpoint();
  RdmaControlPlaneServer server(endpoint);
  server.Start();
  RdmaControlPlaneClient client(endpoint);

  client.PublishServerReady(0);
  client.SendHeartbeat(0);
  const RdmaServerHealthSnapshot snapshot = client.GetServerHealth(0, -1);
  EXPECT_TRUE(snapshot.known);
  client.WaitServer(0, 500);
  client.SendHeartbeat(0);

  // Different RPC kinds on one lazily opened, shared channel.
  EXPECT_EQ(HealthOf(server, 0), RdmaServerHealth::kServing);

  server.Stop();
}

} // namespace
} // namespace petps
