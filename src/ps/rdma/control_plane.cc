#include "ps/rdma/control_plane.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rdma_control_plane.grpc.pb.h"

namespace petps {
namespace {

using recstoreps::rdma::GetMetaRequest;
using recstoreps::rdma::GetMetaResponse;
using recstoreps::rdma::ProbeRequest;
using recstoreps::rdma::ProbeResponse;
using recstoreps::rdma::PublishMetaRequest;
using recstoreps::rdma::PublishMetaResponse;
using recstoreps::rdma::PublishServerDrainingRequest;
using recstoreps::rdma::PublishServerDrainingResponse;
using recstoreps::rdma::PublishServerReadyRequest;
using recstoreps::rdma::PublishServerReadyResponse;
using recstoreps::rdma::RawVerbsMetadata;
using recstoreps::rdma::RdmaControlPlane;
using recstoreps::rdma::WaitServerReadyRequest;
using recstoreps::rdma::WaitServerReadyResponse;
using recstoreps::rdma::WaitServerRequest;
using recstoreps::rdma::WaitServerResponse;

std::string EndpointString(const RdmaControlPlaneEndpoint& endpoint) {
  return endpoint.host + ":" + std::to_string(endpoint.port);
}

std::chrono::system_clock::time_point DeadlineFromNow(int timeout_ms) {
  return std::chrono::system_clock::now() +
         std::chrono::milliseconds(timeout_ms);
}

std::string GrpcStatusText(const grpc::Status& status) {
  if (status.error_message().empty()) {
    return status.error_code() == grpc::StatusCode::OK
             ? std::string("OK")
             : std::to_string(status.error_code());
  }
  return status.error_message();
}

void ThrowIfNotOk(const grpc::Status& status, const std::string& operation) {
  if (status.ok()) {
    return;
  }
  throw std::runtime_error(
      "control-plane " + operation + " failed: " + GrpcStatusText(status));
}

std::string EncodeMetaBytes(const RawVerbsNodeMeta& meta) {
  return EncodeRawVerbsNodeMeta(meta);
}

RawVerbsNodeMeta DecodeMetaBytes(const std::string& payload) {
  return DecodeRawVerbsNodeMeta(payload);
}

grpc::Status MakeDeadlineExceeded(const std::string& message) {
  return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, message);
}

grpc::Status MakeUnavailable(const std::string& message) {
  return grpc::Status(grpc::StatusCode::UNAVAILABLE, message);
}

} // namespace

std::string EncodeRawVerbsNodeMeta(const RawVerbsNodeMeta& meta) {
  RawVerbsMetadata wire;
  wire.set_magic(kRawVerbsMetadataMagic);
  wire.set_protocol_version(meta.protocol_version);
  wire.set_deployment_id(meta.deployment_id);
  wire.set_deployment_epoch(meta.deployment_epoch);
  wire.set_configuration_digest(meta.configuration_digest);
  wire.set_fabric_digest(meta.fabric_digest);
  wire.set_node_id(meta.node_id);
  wire.set_logical_id(meta.logical_id);
  wire.set_node_role(meta.node_role);
  wire.set_lid(meta.lid);
  wire.set_qpn(meta.qpn);
  wire.set_psn(meta.psn);
  wire.set_rkey(meta.rkey);
  wire.set_base_addr(meta.base_addr);
  wire.set_gid(meta.gid, sizeof(meta.gid));
  wire.set_port_num(meta.port_num);
  wire.set_link_layer(meta.link_layer);
  wire.set_gid_index(meta.gid_index);
  wire.set_active_mtu(meta.active_mtu);
  wire.set_fabric_mode(meta.fabric_mode);
  std::string payload;
  if (!wire.SerializeToString(&payload)) {
    throw std::runtime_error("failed to serialize RawVerbsNodeMeta");
  }
  return payload;
}

RawVerbsNodeMeta DecodeRawVerbsNodeMeta(const std::string& payload) {
  RawVerbsMetadata wire;
  if (payload.empty() || payload.size() > kMaxRawVerbsMetadataBytes ||
      !wire.ParseFromString(payload) ||
      wire.magic() != kRawVerbsMetadataMagic ||
      wire.protocol_version() != recstore::kRdmaDeploymentProtocolVersion ||
      wire.deployment_id().empty() || wire.deployment_epoch() == 0 ||
      wire.configuration_digest().empty() || wire.fabric_digest().empty() ||
      wire.node_id() < 0 || wire.node_id() > UINT16_MAX ||
      wire.node_role() >
          static_cast<std::uint32_t>(recstore::RdmaNodeRole::kClient) ||
      wire.lid() > UINT16_MAX || wire.gid().size() != 16 ||
      wire.port_num() == 0 || wire.port_num() > UINT8_MAX ||
      wire.gid_index() < 0 || wire.active_mtu() < IBV_MTU_256 ||
      wire.active_mtu() > IBV_MTU_4096 ||
      (wire.link_layer() != IBV_LINK_LAYER_INFINIBAND &&
       wire.link_layer() != IBV_LINK_LAYER_ETHERNET) ||
      (wire.fabric_mode() ==
           static_cast<std::uint32_t>(recstore::RdmaFabricMode::kIb) &&
       wire.link_layer() != IBV_LINK_LAYER_INFINIBAND) ||
      (wire.fabric_mode() !=
           static_cast<std::uint32_t>(recstore::RdmaFabricMode::kIb) &&
       wire.link_layer() != IBV_LINK_LAYER_ETHERNET) ||
      wire.fabric_mode() >
          static_cast<std::uint32_t>(recstore::RdmaFabricMode::kRocEv2)) {
    throw std::runtime_error("invalid versioned RawVerbsNodeMeta payload");
  }
  RawVerbsNodeMeta meta{};
  meta.protocol_version     = wire.protocol_version();
  meta.deployment_id        = wire.deployment_id();
  meta.deployment_epoch     = wire.deployment_epoch();
  meta.configuration_digest = wire.configuration_digest();
  meta.fabric_digest        = wire.fabric_digest();
  meta.node_id              = static_cast<std::uint16_t>(wire.node_id());
  meta.logical_id           = wire.logical_id();
  meta.node_role            = static_cast<std::uint8_t>(wire.node_role());
  meta.lid                  = static_cast<std::uint16_t>(wire.lid());
  meta.qpn                  = wire.qpn();
  meta.psn                  = wire.psn();
  meta.rkey                 = wire.rkey();
  meta.base_addr            = wire.base_addr();
  std::memcpy(meta.gid, wire.gid().data(), sizeof(meta.gid));
  meta.port_num    = static_cast<std::uint8_t>(wire.port_num());
  meta.link_layer  = static_cast<std::uint8_t>(wire.link_layer());
  meta.gid_index   = wire.gid_index();
  meta.active_mtu  = static_cast<std::uint8_t>(wire.active_mtu());
  meta.fabric_mode = static_cast<std::uint8_t>(wire.fabric_mode());
  return meta;
}

void ValidateRawVerbsPeerMeta(const RawVerbsConfig& local,
                              int expected_node_id,
                              const RawVerbsNodeMeta& remote) {
  const bool expected_server = expected_node_id < local.num_servers;
  const int expected_logical_id =
      expected_server ? expected_node_id : expected_node_id - local.num_servers;
  const auto expected_role = static_cast<std::uint8_t>(
      expected_server ? recstore::RdmaNodeRole::kServer
                      : recstore::RdmaNodeRole::kClient);
  const auto expected_link_layer = static_cast<std::uint8_t>(
      local.fabric_mode == recstore::RdmaFabricMode::kIb
          ? IBV_LINK_LAYER_INFINIBAND
          : IBV_LINK_LAYER_ETHERNET);
  const auto incompatible = [expected_node_id](const char* field) {
    throw std::runtime_error(
        "incompatible RDMA peer metadata field=" + std::string(field) +
        " node_id=" + std::to_string(expected_node_id));
  };
  if (remote.node_id != expected_node_id)
    incompatible("node_id");
  if (remote.logical_id != expected_logical_id)
    incompatible("logical_id");
  if (remote.node_role != expected_role)
    incompatible("node_role");
  if (remote.protocol_version != local.protocol_version)
    incompatible("protocol_version");
  if (remote.deployment_id != local.deployment_id)
    incompatible("deployment_id");
  if (remote.deployment_epoch != local.deployment_epoch)
    incompatible("deployment_epoch");
  if (remote.configuration_digest != local.configuration_digest)
    incompatible("configuration_digest");
  if (remote.fabric_digest != local.fabric_digest)
    incompatible("fabric_digest");
  if (remote.link_layer != expected_link_layer)
    incompatible("link_layer");
  if (remote.fabric_mode != static_cast<std::uint8_t>(local.fabric_mode))
    incompatible("fabric_mode");
  if (remote.port_num == 0)
    incompatible("port_num");
  if (remote.gid_index < 0)
    incompatible("gid_index");
  if (remote.active_mtu == 0)
    incompatible("active_mtu");
}

class RdmaControlPlaneService final : public RdmaControlPlane::Service {
public:
  explicit RdmaControlPlaneService(RdmaControlPlaneServer* owner)
      : owner_(owner) {}

  grpc::Status ValidateDeploymentIdentity(
      const std::string& deployment_id, std::uint64_t deployment_epoch) const {
    if (deployment_id.empty() || deployment_epoch == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "control-plane deployment identity is required");
    }
    if (deployment_id != owner_->endpoint_.deployment_id ||
        deployment_epoch != owner_->endpoint_.deployment_epoch) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "control-plane deployment identity mismatch");
    }
    return grpc::Status::OK;
  }

  grpc::Status PublishMeta(grpc::ServerContext*,
                           const PublishMetaRequest* request,
                           PublishMetaResponse*) override {
    const grpc::Status identity = ValidateDeploymentIdentity(
        request->deployment_id(), request->deployment_epoch());
    if (!identity.ok()) {
      return identity;
    }
    if (request->publisher_node_id() < 0 || request->publisher_lane() < 0 ||
        request->receiver_node_id() < 0 || request->receiver_lane() < 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "metadata node and lane ids must be non-negative");
    }
    const RdmaControlPlaneServer::MetaKey key{
        request->deployment_id(),
        request->deployment_epoch(),
        request->publisher_node_id(),
        request->publisher_lane(),
        request->receiver_node_id(),
        request->receiver_lane(),
    };
    RawVerbsNodeMeta meta;
    try {
      meta = DecodeMetaBytes(request->meta());
    } catch (const std::exception& error) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error.what());
    }
    if (meta.protocol_version != owner_->endpoint_.protocol_version ||
        meta.configuration_digest != owner_->endpoint_.configuration_digest ||
        meta.fabric_digest != owner_->endpoint_.fabric_digest ||
        meta.deployment_id != request->deployment_id() ||
        meta.deployment_epoch != request->deployment_epoch() ||
        meta.node_id != request->publisher_node_id()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "metadata deployment identity mismatch");
    }
    {
      std::lock_guard<std::mutex> guard(owner_->mu_);
      owner_->metadata_[key] = meta;
    }
    owner_->cv_.notify_all();
    return grpc::Status::OK;
  }

  grpc::Status GetMeta(grpc::ServerContext*,
                       const GetMetaRequest* request,
                       GetMetaResponse* response) override {
    const grpc::Status identity = ValidateDeploymentIdentity(
        request->deployment_id(), request->deployment_epoch());
    if (!identity.ok()) {
      return identity;
    }
    if (request->publisher_node_id() < 0 || request->publisher_lane() < 0 ||
        request->receiver_node_id() < 0 || request->receiver_lane() < 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "metadata node and lane ids must be non-negative");
    }
    const RdmaControlPlaneServer::MetaKey key{
        request->deployment_id(),
        request->deployment_epoch(),
        request->publisher_node_id(),
        request->publisher_lane(),
        request->receiver_node_id(),
        request->receiver_lane(),
    };
    const int timeout_ms =
        request->timeout_ms() > 0
            ? request->timeout_ms()
            : owner_->endpoint_.timeout_ms;
    std::unique_lock<std::mutex> lock(owner_->mu_);
    const bool ready =
        owner_->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
          return owner_->stop_requested_.load(std::memory_order_relaxed) ||
                 owner_->metadata_.find(key) != owner_->metadata_.end();
        });
    if (!ready) {
      return MakeDeadlineExceeded(
          "get_meta timeout key=" + std::to_string(key.publisher_node_id) +
          ":" + std::to_string(key.publisher_lane) + "->" +
          std::to_string(key.receiver_node_id) + ":" +
          std::to_string(key.receiver_lane));
    }
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    response->set_meta(EncodeMetaBytes(owner_->metadata_.at(key)));
    return grpc::Status::OK;
  }

  grpc::Status PublishServerReady(grpc::ServerContext*,
                                  const PublishServerReadyRequest* request,
                                  PublishServerReadyResponse*) override {
    const grpc::Status identity = ValidateDeploymentIdentity(
        request->deployment_id(), request->deployment_epoch());
    if (!identity.ok()) {
      return identity;
    }
    if (request->protocol_version() != owner_->endpoint_.protocol_version ||
        request->configuration_digest().empty() ||
        request->fabric_digest().empty() ||
        request->configuration_digest() !=
            owner_->endpoint_.configuration_digest ||
        request->fabric_digest() != owner_->endpoint_.fabric_digest ||
        request->server_id() < 0 ||
        (owner_->endpoint_.num_servers > 0 &&
         request->server_id() >= owner_->endpoint_.num_servers)) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "invalid versioned server READY record");
    }
    {
      std::lock_guard<std::mutex> guard(owner_->mu_);
      auto& record = owner_->ready_servers_[{request->deployment_id(),
                                             request->deployment_epoch(),
                                             request->server_id()}];
      if (record.state ==
          RdmaControlPlaneServer::ReadyRecord::State::kDraining) {
        return grpc::Status(
            grpc::StatusCode::FAILED_PRECONDITION,
            "draining server cannot become ready in same epoch");
      }
      record = {request->protocol_version(),
                request->configuration_digest(),
                request->fabric_digest(),
                RdmaControlPlaneServer::ReadyRecord::State::kServing};
    }
    owner_->cv_.notify_all();
    return grpc::Status::OK;
  }

  grpc::Status PublishServerDraining(
      grpc::ServerContext*,
      const PublishServerDrainingRequest* request,
      PublishServerDrainingResponse*) override {
    const grpc::Status identity = ValidateDeploymentIdentity(
        request->deployment_id(), request->deployment_epoch());
    if (!identity.ok()) {
      return identity;
    }
    if (request->protocol_version() != owner_->endpoint_.protocol_version ||
        request->configuration_digest() !=
            owner_->endpoint_.configuration_digest ||
        request->fabric_digest() != owner_->endpoint_.fabric_digest ||
        request->server_id() < 0 ||
        (owner_->endpoint_.num_servers > 0 &&
         request->server_id() >= owner_->endpoint_.num_servers)) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "invalid versioned server DRAINING record");
    }
    {
      std::lock_guard<std::mutex> guard(owner_->mu_);
      const RdmaControlPlaneServer::ReadyKey key{
          request->deployment_id(),
          request->deployment_epoch(),
          request->server_id()};
      const auto it = owner_->ready_servers_.find(key);
      if (it == owner_->ready_servers_.end()) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "server must be ready before draining");
      }
      if (it->second.protocol_version != request->protocol_version() ||
          it->second.configuration_digest != request->configuration_digest() ||
          it->second.fabric_digest != request->fabric_digest()) {
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "server DRAINING deployment contract mismatch");
      }
      it->second.state = RdmaControlPlaneServer::ReadyRecord::State::kDraining;
    }
    owner_->cv_.notify_all();
    return grpc::Status::OK;
  }

  grpc::Status WaitServer(grpc::ServerContext*,
                          const WaitServerRequest* request,
                          WaitServerResponse*) override {
    const grpc::Status identity = ValidateDeploymentIdentity(
        request->deployment_id(), request->deployment_epoch());
    if (!identity.ok()) {
      return identity;
    }
    if (request->protocol_version() != owner_->endpoint_.protocol_version ||
        request->configuration_digest() !=
            owner_->endpoint_.configuration_digest ||
        request->fabric_digest() != owner_->endpoint_.fabric_digest ||
        request->server_id() < 0 ||
        (owner_->endpoint_.num_servers > 0 &&
         request->server_id() >= owner_->endpoint_.num_servers)) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "server READY deployment contract mismatch");
    }
    const int timeout_ms =
        request->timeout_ms() > 0
            ? request->timeout_ms()
            : owner_->endpoint_.timeout_ms;
    const RdmaControlPlaneServer::ReadyKey key{
        request->deployment_id(),
        request->deployment_epoch(),
        request->server_id()};
    std::unique_lock<std::mutex> lock(owner_->mu_);
    const bool ready =
        owner_->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
          return owner_->stop_requested_.load(std::memory_order_relaxed) ||
                 owner_->ready_servers_.find(key) !=
                     owner_->ready_servers_.end();
        });
    if (!ready) {
      return MakeDeadlineExceeded("wait_server timeout server_id=" +
                                  std::to_string(request->server_id()));
    }
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    const auto& record = owner_->ready_servers_.at(key);
    if (record.protocol_version != request->protocol_version() ||
        record.configuration_digest != request->configuration_digest() ||
        record.fabric_digest != request->fabric_digest()) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "server READY deployment contract mismatch");
    }
    if (record.state == RdmaControlPlaneServer::ReadyRecord::State::kDraining) {
      return MakeUnavailable(
          "server draining server_id=" + std::to_string(request->server_id()));
    }
    return grpc::Status::OK;
  }

  grpc::Status WaitServerReady(grpc::ServerContext*,
                               const WaitServerReadyRequest* request,
                               WaitServerReadyResponse*) override {
    const grpc::Status identity = ValidateDeploymentIdentity(
        request->deployment_id(), request->deployment_epoch());
    if (!identity.ok()) {
      return identity;
    }
    if (request->protocol_version() != owner_->endpoint_.protocol_version ||
        request->configuration_digest() !=
            owner_->endpoint_.configuration_digest ||
        request->fabric_digest() != owner_->endpoint_.fabric_digest ||
        request->num_servers() <= 0 ||
        (owner_->endpoint_.num_servers > 0 &&
         request->num_servers() > owner_->endpoint_.num_servers)) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "server READY deployment contract mismatch");
    }
    const int timeout_ms =
        request->timeout_ms() > 0
            ? request->timeout_ms()
            : owner_->endpoint_.timeout_ms;
    std::unique_lock<std::mutex> lock(owner_->mu_);
    const bool ready =
        owner_->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
          return owner_->stop_requested_.load(std::memory_order_relaxed) ||
                 owner_->ReadyCount(
                     request->deployment_id(),
                     request->deployment_epoch(),
                     request->protocol_version(),
                     request->configuration_digest(),
                     request->fabric_digest()) >= request->num_servers() ||
                 owner_->HasReadyContractMismatch(
                     request->deployment_id(),
                     request->deployment_epoch(),
                     request->protocol_version(),
                     request->configuration_digest(),
                     request->fabric_digest()) ||
                 owner_->HasDrainingServer(request->deployment_id(),
                                           request->deployment_epoch(),
                                           request->num_servers());
        });
    if (owner_->HasReadyContractMismatch(
            request->deployment_id(),
            request->deployment_epoch(),
            request->protocol_version(),
            request->configuration_digest(),
            request->fabric_digest())) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "server READY deployment contract mismatch");
    }
    if (owner_->HasDrainingServer(request->deployment_id(),
                                  request->deployment_epoch(),
                                  request->num_servers())) {
      return MakeUnavailable("one or more RDMA servers are draining");
    }
    if (!ready) {
      return MakeDeadlineExceeded(
          "wait_server_ready timeout ready=" +
          std::to_string(owner_->ReadyCount(
              request->deployment_id(),
              request->deployment_epoch(),
              request->protocol_version(),
              request->configuration_digest(),
              request->fabric_digest())) +
          "/" + std::to_string(request->num_servers()));
    }
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    return grpc::Status::OK;
  }

  grpc::Status
  Probe(grpc::ServerContext*, const ProbeRequest*, ProbeResponse*) override {
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    return grpc::Status::OK;
  }

private:
  RdmaControlPlaneServer* owner_;
};

RdmaControlPlaneClient::RdmaControlPlaneClient(
    RdmaControlPlaneEndpoint endpoint)
    : endpoint_(std::move(endpoint)) {}

void RdmaControlPlaneClient::PublishMeta(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane,
    const RawVerbsNodeMeta& meta) const {
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  PublishMetaRequest request;
  request.set_publisher_node_id(publisher_node_id);
  request.set_publisher_lane(publisher_lane);
  request.set_receiver_node_id(receiver_node_id);
  request.set_receiver_lane(receiver_lane);
  request.set_meta(EncodeMetaBytes(meta));
  request.set_deployment_id(endpoint_.deployment_id);
  request.set_deployment_epoch(endpoint_.deployment_epoch);

  PublishMetaResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(endpoint_.timeout_ms));
  ThrowIfNotOk(stub->PublishMeta(&context, request, &response), "publish_meta");
}

RawVerbsNodeMeta RdmaControlPlaneClient::GetMeta(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane,
    int timeout_ms) const {
  const int effective_timeout_ms =
      timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms;
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  GetMetaRequest request;
  request.set_publisher_node_id(publisher_node_id);
  request.set_publisher_lane(publisher_lane);
  request.set_receiver_node_id(receiver_node_id);
  request.set_receiver_lane(receiver_lane);
  request.set_timeout_ms(effective_timeout_ms);
  request.set_deployment_id(endpoint_.deployment_id);
  request.set_deployment_epoch(endpoint_.deployment_epoch);

  GetMetaResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(effective_timeout_ms));
  ThrowIfNotOk(stub->GetMeta(&context, request, &response), "get_meta");
  return DecodeMetaBytes(response.meta());
}

void RdmaControlPlaneClient::PublishServerReady(int server_id) const {
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  PublishServerReadyRequest request;
  request.set_server_id(server_id);
  request.set_protocol_version(endpoint_.protocol_version);
  request.set_deployment_id(endpoint_.deployment_id);
  request.set_deployment_epoch(endpoint_.deployment_epoch);
  request.set_configuration_digest(endpoint_.configuration_digest);
  request.set_fabric_digest(endpoint_.fabric_digest);

  PublishServerReadyResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(endpoint_.timeout_ms));
  ThrowIfNotOk(stub->PublishServerReady(&context, request, &response),
               "server_ready");
}

void RdmaControlPlaneClient::PublishServerDraining(int server_id) const {
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  PublishServerDrainingRequest request;
  request.set_server_id(server_id);
  request.set_protocol_version(endpoint_.protocol_version);
  request.set_deployment_id(endpoint_.deployment_id);
  request.set_deployment_epoch(endpoint_.deployment_epoch);
  request.set_configuration_digest(endpoint_.configuration_digest);
  request.set_fabric_digest(endpoint_.fabric_digest);

  PublishServerDrainingResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(endpoint_.timeout_ms));
  ThrowIfNotOk(stub->PublishServerDraining(&context, request, &response),
               "server_draining");
}

void RdmaControlPlaneClient::WaitServer(int server_id, int timeout_ms) const {
  const int effective_timeout_ms =
      timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms;
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  WaitServerRequest request;
  request.set_server_id(server_id);
  request.set_timeout_ms(effective_timeout_ms);
  request.set_protocol_version(endpoint_.protocol_version);
  request.set_deployment_id(endpoint_.deployment_id);
  request.set_deployment_epoch(endpoint_.deployment_epoch);
  request.set_configuration_digest(endpoint_.configuration_digest);
  request.set_fabric_digest(endpoint_.fabric_digest);

  WaitServerResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(effective_timeout_ms));
  ThrowIfNotOk(stub->WaitServer(&context, request, &response), "wait_server");
}

void RdmaControlPlaneClient::WaitServerReady(int num_servers,
                                             int timeout_ms) const {
  const int effective_timeout_ms =
      timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms;
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  WaitServerReadyRequest request;
  request.set_num_servers(num_servers);
  request.set_timeout_ms(effective_timeout_ms);
  request.set_protocol_version(endpoint_.protocol_version);
  request.set_deployment_id(endpoint_.deployment_id);
  request.set_deployment_epoch(endpoint_.deployment_epoch);
  request.set_configuration_digest(endpoint_.configuration_digest);
  request.set_fabric_digest(endpoint_.fabric_digest);

  WaitServerReadyResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(effective_timeout_ms));
  ThrowIfNotOk(stub->WaitServerReady(&context, request, &response),
               "wait_server_ready");
}

RdmaControlPlaneServer::RdmaControlPlaneServer(
    RdmaControlPlaneEndpoint endpoint)
    : endpoint_(std::move(endpoint)) {}

RdmaControlPlaneServer::~RdmaControlPlaneServer() { Stop(); }

std::size_t
RdmaControlPlaneServer::MetaKeyHash::operator()(const MetaKey& key) const {
  std::size_t hash = std::hash<std::string>{}(key.deployment_id);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.deployment_epoch);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.publisher_node_id);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.publisher_lane);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.receiver_node_id);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.receiver_lane);
  return hash;
}

std::size_t
RdmaControlPlaneServer::ReadyKeyHash::operator()(const ReadyKey& key) const {
  std::size_t hash = std::hash<std::string>{}(key.deployment_id);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.deployment_epoch);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.server_id);
  return hash;
}

int RdmaControlPlaneServer::ReadyCount(
    const std::string& deployment_id,
    std::uint64_t deployment_epoch,
    std::uint32_t protocol_version,
    const std::string& configuration_digest,
    const std::string& fabric_digest) const {
  int count = 0;
  for (const auto& entry : ready_servers_) {
    if (entry.first.deployment_id == deployment_id &&
        entry.first.deployment_epoch == deployment_epoch &&
        entry.second.state == ReadyRecord::State::kServing &&
        entry.second.protocol_version == protocol_version &&
        entry.second.configuration_digest == configuration_digest &&
        entry.second.fabric_digest == fabric_digest) {
      ++count;
    }
  }
  return count;
}

bool RdmaControlPlaneServer::HasReadyContractMismatch(
    const std::string& deployment_id,
    std::uint64_t deployment_epoch,
    std::uint32_t protocol_version,
    const std::string& configuration_digest,
    const std::string& fabric_digest) const {
  for (const auto& entry : ready_servers_) {
    if (entry.first.deployment_id == deployment_id &&
        entry.first.deployment_epoch == deployment_epoch &&
        (entry.second.protocol_version != protocol_version ||
         entry.second.configuration_digest != configuration_digest ||
         entry.second.fabric_digest != fabric_digest)) {
      return true;
    }
  }
  return false;
}

bool RdmaControlPlaneServer::HasDrainingServer(
    const std::string& deployment_id,
    std::uint64_t deployment_epoch,
    int num_servers) const {
  for (const auto& entry : ready_servers_) {
    if (entry.first.deployment_id == deployment_id &&
        entry.first.deployment_epoch == deployment_epoch &&
        entry.first.server_id >= 0 && entry.first.server_id < num_servers &&
        entry.second.state == ReadyRecord::State::kDraining) {
      return true;
    }
  }
  return false;
}

void RdmaControlPlaneServer::Start() {
  if (server_ != nullptr) {
    return;
  }
  if (endpoint_.host.empty() || endpoint_.port < 1 || endpoint_.port > 65535 ||
      endpoint_.timeout_ms <= 0 || endpoint_.deployment_id.empty() ||
      endpoint_.deployment_epoch == 0 ||
      endpoint_.protocol_version != recstore::kRdmaDeploymentProtocolVersion ||
      endpoint_.configuration_digest.empty() ||
      endpoint_.fabric_digest.empty()) {
    throw std::invalid_argument("invalid RDMA control-plane endpoint contract");
  }
  stop_requested_.store(false, std::memory_order_relaxed);
  service_ = std::make_unique<RdmaControlPlaneService>(this);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(
      EndpointString(endpoint_), grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  if (server_ == nullptr) {
    throw std::runtime_error("control-plane gRPC server failed to listen on " +
                             EndpointString(endpoint_));
  }
}

void RdmaControlPlaneServer::Stop() {
  if (server_ == nullptr) {
    return;
  }
  stop_requested_.store(true, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> guard(mu_);
    cv_.notify_all();
  }
  server_->Shutdown();
  server_.reset();
  service_.reset();
}

} // namespace petps
