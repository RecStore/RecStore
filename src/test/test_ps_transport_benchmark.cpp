#include <gtest/gtest.h>

#include "benchmark/ps/ps_transport_benchmark_config.h"

TEST(PSTransportBenchmarkConfigTest, RejectsUnknownTransport) {
  EXPECT_THROW(
      {
        const auto transport = NormalizeBenchmarkTransport("banana");
        (void)transport;
      },
      std::invalid_argument);
}

TEST(PSTransportBenchmarkConfigTest, BuildsRpcConfig) {
  auto config = BuildRpcBenchmarkConfig("BRPC", "127.0.0.1", 25000);
  EXPECT_EQ(config["cache_ps"]["ps_type"], "BRPC");
  EXPECT_EQ(config["client"]["host"], "127.0.0.1");
  EXPECT_EQ(config["client"]["port"], 25000);
}

TEST(PSTransportBenchmarkConfigTest,
     RequiresManagedReceiveBufferOnlyForSingleShardRdma) {
  EXPECT_TRUE(BenchmarkRequiresManagedReceiveBuffer("rdma", 1));
  EXPECT_FALSE(BenchmarkRequiresManagedReceiveBuffer("rdma", 2));
  EXPECT_FALSE(BenchmarkRequiresManagedReceiveBuffer("grpc", 1));
  EXPECT_FALSE(BenchmarkRequiresManagedReceiveBuffer("brpc", 1));
}

TEST(PSTransportBenchmarkConfigTest, UsesVectorGetForBrpcOnly) {
  EXPECT_TRUE(BenchmarkUsesVectorGet("brpc"));
  EXPECT_FALSE(BenchmarkUsesVectorGet("grpc"));
  EXPECT_FALSE(BenchmarkUsesVectorGet("rdma"));
}

TEST(PSTransportBenchmarkConfigTest, UsesFlatGetForLocalShmOnly) {
  EXPECT_TRUE(BenchmarkUsesFlatGet("local_shm"));
  EXPECT_FALSE(BenchmarkUsesFlatGet("grpc"));
  EXPECT_FALSE(BenchmarkUsesFlatGet("brpc"));
  EXPECT_FALSE(BenchmarkUsesFlatGet("rdma"));
}

TEST(PSTransportBenchmarkConfigTest, WriteReturnSemanticsMatchTransport) {
  EXPECT_TRUE(BenchmarkWriteReturnsZeroOnSuccess("grpc"));
  EXPECT_TRUE(BenchmarkWriteReturnsZeroOnSuccess("brpc"));
  EXPECT_TRUE(BenchmarkWriteReturnsZeroOnSuccess("rdma"));
  EXPECT_TRUE(BenchmarkWriteReturnsZeroOnSuccess("local_shm"));

  EXPECT_TRUE(BenchmarkWriteSucceeded("grpc", 0));
  EXPECT_TRUE(BenchmarkWriteSucceeded("brpc", 0));
  EXPECT_TRUE(BenchmarkWriteSucceeded("rdma", 0));
  EXPECT_TRUE(BenchmarkWriteSucceeded("local_shm", 0));

  EXPECT_FALSE(BenchmarkWriteSucceeded("grpc", 1));
  EXPECT_FALSE(BenchmarkWriteSucceeded("brpc", 1));
  EXPECT_FALSE(BenchmarkWriteSucceeded("rdma", 1));
  EXPECT_FALSE(BenchmarkWriteSucceeded("local_shm", 1));
}

TEST(PSTransportBenchmarkConfigTest,
     WriteReturnSemanticsCanOverrideDistributedRpcClients) {
  EXPECT_TRUE(BenchmarkWriteSucceeded("grpc", 0, true));
  EXPECT_TRUE(BenchmarkWriteSucceeded("brpc", 0, true));
  EXPECT_FALSE(BenchmarkWriteSucceeded("grpc", 1, true));
  EXPECT_FALSE(BenchmarkWriteSucceeded("brpc", 1, true));
}

TEST(PSTransportBenchmarkConfigTest, ReadReturnSemanticsMatchTransport) {
  EXPECT_TRUE(BenchmarkReadReturnsZeroOnSuccess("grpc"));
  EXPECT_TRUE(BenchmarkReadReturnsZeroOnSuccess("brpc"));
  EXPECT_TRUE(BenchmarkReadReturnsZeroOnSuccess("rdma"));
  EXPECT_TRUE(BenchmarkReadReturnsZeroOnSuccess("local_shm"));

  EXPECT_TRUE(BenchmarkReadSucceeded("grpc", 0));
  EXPECT_TRUE(BenchmarkReadSucceeded("brpc", 0));
  EXPECT_TRUE(BenchmarkReadSucceeded("rdma", 0));
  EXPECT_TRUE(BenchmarkReadSucceeded("local_shm", 0));

  EXPECT_FALSE(BenchmarkReadSucceeded("grpc", 1));
  EXPECT_FALSE(BenchmarkReadSucceeded("brpc", 1));
  EXPECT_FALSE(BenchmarkReadSucceeded("rdma", 1));
  EXPECT_FALSE(BenchmarkReadSucceeded("local_shm", 1));
}

TEST(PSTransportBenchmarkConfigTest,
     ReadReturnSemanticsCanOverrideDistributedRpcClients) {
  EXPECT_TRUE(BenchmarkReadSucceeded("grpc", 0, true));
  EXPECT_TRUE(BenchmarkReadSucceeded("brpc", 0, true));
  EXPECT_FALSE(BenchmarkReadSucceeded("grpc", 1, true));
  EXPECT_FALSE(BenchmarkReadSucceeded("brpc", 1, true));
}
