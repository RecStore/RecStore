#include <gtest/gtest.h>

#include "ps/rdma/raw_verbs_transport.h"

TEST(RawVerbsRegionAllocatorTest,
     SkipsReservedRegionWhenAllocationWouldOverlap) {
  petps::RawVerbsRegionAllocator allocator(1024);
  allocator.SetReservedRegion({256, 128});

  EXPECT_EQ(allocator.Allocate(64), 0);
  EXPECT_EQ(allocator.Allocate(192), 64);
  EXPECT_EQ(allocator.Allocate(64), 384);
}

TEST(RawVerbsRegionAllocatorTest, ScopeRestoresAllocationOffset) {
  petps::RawVerbsRegionAllocator allocator(1024);
  allocator.SetReservedRegion({256, 128});

  EXPECT_EQ(allocator.current_offset(), 0);
  {
    petps::RawVerbsRegionAllocatorScope scope(&allocator);
    EXPECT_EQ(allocator.Allocate(64), 0);
    EXPECT_EQ(allocator.Allocate(64), 64);
    EXPECT_EQ(allocator.current_offset(), 128);
  }
  EXPECT_EQ(allocator.current_offset(), 0);
  EXPECT_EQ(allocator.Allocate(64), 0);
}

TEST(RawVerbsRegionAllocatorTest, RejectsReservedRegionOutsideLocalMemory) {
  petps::RawVerbsRegionAllocator allocator(512);
  EXPECT_THROW(allocator.SetReservedRegion({448, 128}), std::runtime_error);
}

TEST(RawVerbsEndpointTest, MetaKeyIncludesLocalAndRemoteLanes) {
  EXPECT_EQ(petps::RawVerbsMetaKey(1, 2, 3, 4),
            "raw-verbs-meta-1-lane-2-to-3-lane-4");
}

TEST(RawVerbsEndpointTest, PeerFilterCanConnectOnlyServersOrClients) {
  petps::RawVerbsConfig server_endpoint;
  server_endpoint.num_servers        = 2;
  server_endpoint.num_clients        = 3;
  server_endpoint.connect_to_servers = false;
  server_endpoint.connect_to_clients = true;

  EXPECT_FALSE(petps::ShouldRawVerbsConnectToNode(server_endpoint, 0));
  EXPECT_FALSE(petps::ShouldRawVerbsConnectToNode(server_endpoint, 1));
  EXPECT_TRUE(petps::ShouldRawVerbsConnectToNode(server_endpoint, 2));
  EXPECT_TRUE(petps::ShouldRawVerbsConnectToNode(server_endpoint, 4));

  petps::RawVerbsConfig client_endpoint = server_endpoint;
  client_endpoint.global_id             = 2;
  client_endpoint.connect_to_servers    = true;
  client_endpoint.connect_to_clients    = false;

  EXPECT_TRUE(petps::ShouldRawVerbsConnectToNode(client_endpoint, 0));
  EXPECT_TRUE(petps::ShouldRawVerbsConnectToNode(client_endpoint, 1));
  EXPECT_FALSE(petps::ShouldRawVerbsConnectToNode(client_endpoint, 2));
  EXPECT_FALSE(petps::ShouldRawVerbsConnectToNode(client_endpoint, 4));
}
