#include "Postoffice.h"

#include "mayfly_config.h"
#include "base/log.h"

#include <cstdlib>
#include <libmemcached/memcached.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <utility>

DEFINE_int32(num_server_processes, 1, "# of server processes");
DEFINE_int32(num_client_processes, 1, "# of client processes");
DEFINE_int32(global_id, 0, "");

static std::string trim(const std::string& s) {
  std::string res = s;
  if (!res.empty()) {
    res.erase(0, res.find_first_not_of(" "));
    res.erase(res.find_last_not_of(" ") + 1);
  }
  return res;
}

static std::pair<std::string, std::string> ResolveMemcachedEndpoint() {
  const char* env_host = std::getenv("RECSTORE_MEMCACHED_HOST");
  const char* env_port = std::getenv("RECSTORE_MEMCACHED_PORT");
  if (env_host != nullptr && env_port != nullptr) {
    return {env_host, env_port};
  }

  std::ifstream conf(MAYFLY_PATH "/memcached.conf");
  if (!conf) {
    LOG(FATAL) << "can't open memcached.conf";
  }

  std::string addr;
  std::string port;
  std::getline(conf, addr);
  std::getline(conf, port);
  return {trim(addr), trim(port)};
}

XPostoffice::XPostoffice() {
  num_servers_ = FLAGS_num_server_processes;
  num_clients_ = FLAGS_num_client_processes;

  int g_id = FLAGS_global_id;

  static std::mutex m;
  std::lock_guard<std::mutex> _(m);
  static bool init_ = false;

  CHECK(init_ == false);
  init_ = true;

  global_id_ = g_id;
  if (0 <= g_id && g_id < num_servers_) {
    actor_     = ACTOR_SERVER;
    server_id_ = g_id;
  } else if (num_servers_ <= g_id && g_id < num_servers_ + num_clients_) {
    actor_     = ACTOR_CLIENT;
    client_id_ = g_id - num_servers_;
  } else {
    LOG(FATAL) << "Invalid XPostoffice" << std::endl
               << "global_id = " << global_id_ << std::endl
               << "server_id = " << server_id_ << std::endl
               << "client_id = " << client_id_ << std::endl;
  }
  ConnectMemcached();
}

std::string XPostoffice::MemCachedGet(const std::string& key) {
  size_t l;
  uint32_t flags;
  memcached_return rc;
  char* res;
  while (true) {
    res = memcached_get(memc_, key.c_str(), key.size(), &l, &flags, &rc);
    if (rc == MEMCACHED_SUCCESS) {
      break;
    }
    usleep(400);
  }
  auto ret = std::string(res);
  free(res);
  return ret;
}

void XPostoffice::MemCachedSet(const std::string& key,
                               const std::string& value) {
  memcached_return rc;
  while (true) {
    rc = memcached_set(
        memc_,
        key.c_str(),
        key.size(),
        value.c_str(),
        value.size(),
        (time_t)0,
        (uint32_t)0);
    if (rc == MEMCACHED_SUCCESS) {
      break;
    }
    usleep(400);
  }
}

void XPostoffice::ConnectMemcached() {
  memcached_server_st* servers = NULL;
  memcached_return rc;
  auto [addr, port] = ResolveMemcachedEndpoint();
  std::cout << "use memcached in " << addr << ":" << port << std::endl;

  memc_ = memcached_create(NULL);
  servers =
      memcached_server_list_append(servers, addr.c_str(), std::stoi(port), &rc);
  rc = memcached_server_push(memc_, servers);
}
