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

} // namespace
} // namespace petps
