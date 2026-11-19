#pragma once

#include <stdlib.h>

#include <cmath>
#include <thread>

#include "base/log.h"
#include "base/string.h"
#include "ps/base/shard_manager.h"
#include "storage/kv_engine/base_kv.h"

class LoadDBHelper {
public:
  LoadDBHelper(BaseKV* kv,
               int server_id,
               int warm_thread_count,
               int64_t all_ps_warm_capacity,
               int value_size)
      : kv_(kv),
        server_id_(server_id),
        warm_thread_count_(warm_thread_count),
        warm_capacity_(all_ps_warm_capacity),
        value_size_(value_size) {}

  void PreLoadDB(bool shuffle_load = false) {
    // Both branches stage each row in a fixed 32KB heap buffer.
    CHECK_LE(value_size_, 32 * 1024)
        << "PreLoadDB staging buffer is 32KB; value size is too large";
    uint64_t warm_kv_per_thread =
        (warm_capacity_ + warm_thread_count_ - 1) / warm_thread_count_;
    uint64_t start = 1;
    uint64_t end   = warm_capacity_ + 1;

    // std::atomic<uint64_t> insert_to_this_db{0};
    uint64_t insert_to_this_db = 0;

#pragma omp parallel for reduction(+ : insert_to_this_db)
    for (int64_t j = start; j < end; ++j) {
      if (ShardManager::KeyPartition(j) == server_id_)
        insert_to_this_db++;
    }

    warmup_kv_count_in_this_ps_ = insert_to_this_db;
    LOG(INFO) << base::SFormat(
        "Server{} before load db, warm_capacity_ = {}, {}M, belong to this "
        "partititon inserting {} KVs, "
        "{}M ",
        server_id_,
        warm_capacity_,
        (int)(warm_capacity_ / 1024 / 1024),
        insert_to_this_db,
        (int)(insert_to_this_db / 1024 / 1024));

    if (shuffle_load == false) {
      std::vector<std::thread> th(warm_thread_count_);
      for (int i = 0; i < warm_thread_count_; ++i) {
        th[i] = std::thread([this, i, warm_kv_per_thread, start, end]() {
          char* pool = (char*)malloc(32 * 1024);
          memset(pool, 'e', 32 * 1024);
          for (int64_t j = start + i * warm_kv_per_thread;
               j < std::min(end, start + (i + 1) * warm_kv_per_thread);
               ++j) {
            if (i == 0) {
              RECSTORE_LOG_EVERY_MS(INFO, 60000)
                  << (j - start - i * warm_kv_per_thread) * 100LL /
                         warm_kv_per_thread
                  << " %";
            }
            if (ShardManager::KeyPartition(j) != server_id_)
              continue;

            uint64_t key = j;
            float* emb   = (float*)pool;
            for (int k = 0; k < value_size_ / sizeof(float); k++)
              emb[k] = key;
            kv_->Put(key, std::string_view(pool, value_size_), i);
          }
          free(pool);
        });
      }
      for (int i = 0; i < warm_thread_count_; ++i) {
        th[i].join();
      }
    } else {
      char* pool = (char*)malloc(32 * 1024);
      memset(pool, 'e', 32 * 1024);
      CHECK_EQ(1, warm_thread_count_);

      std::vector<uint64_t> to_insert_keys;
      to_insert_keys.reserve(end - start);
      for (int64_t j = start; j < end; ++j) {
        if (ShardManager::KeyPartition(j) != server_id_)
          continue;
        to_insert_keys.push_back(j);
      }

      for (int64_t j = 0; j < to_insert_keys.size(); ++j) {
        RECSTORE_LOG_EVERY_MS(INFO, 60000)
            << j * 100LL / to_insert_keys.size() << " %";
        uint64_t key = to_insert_keys[j];
        float* emb   = (float*)pool;
        for (int k = 0; k < value_size_ / sizeof(float); k++)
          emb[k] = key;
        kv_->Put(key, std::string_view(pool, value_size_), 0);
      }
      free(pool);
    }
    LOG(INFO) << "server " << server_id_ << " after load db";
  }

  void CheckDBLoad() {
    LOG(INFO) << "server " << server_id_ << " before check db";
    int check_thread_num = 36;
    uint64_t warm_kv_per_thread =
        (warm_capacity_ + check_thread_num - 1) / check_thread_num;
    uint64_t start = 1;
    uint64_t end   = warm_capacity_ + 1;

    std::vector<std::thread> th(check_thread_num);

    std::atomic<uint64_t> inserted_kv_nr{0};

    for (int i = 0; i < check_thread_num; ++i) {
      th[i] = std::thread(
          [this, i, warm_kv_per_thread, start, end, &inserted_kv_nr]() {
            size_t right = 0;
            for (int j = start + i * warm_kv_per_thread;
                 j < std::min(end, start + (i + 1) * warm_kv_per_thread);
                 ++j) {
              if (i == 0) {
                RECSTORE_LOG_EVERY_MS(INFO, 10000)
                    << (j - start - i * warm_kv_per_thread) * 100LL /
                           warm_kv_per_thread
                    << " %";
              }
              uint64_t key = j;
              if (ShardManager::KeyPartition(key) != server_id_)
                continue;
              std::string value;
              kv_->Get(key, value, i);
              float* emb    = (float*)value.c_str();
              bool subright = true;
              for (int k = 0; k < value_size_ / sizeof(float); k++) {
                // CHECK(std::fabs(emb[k] - key) < 1e-6);
                if (!(std::fabs(emb[k] - key) < 1e-6)) {
                  subright = false;
                }
              }

              if (subright)
                right += 1;
            }
            inserted_kv_nr += right;
          });
    }
    for (int i = 0; i < check_thread_num; ++i) {
      th[i].join();
    }

    if (inserted_kv_nr != warmup_kv_count_in_this_ps_)
      LOG(ERROR) << base::SFormat(
          "Only insert {}/{}={}%",
          inserted_kv_nr.load(),
          warmup_kv_count_in_this_ps_,
          inserted_kv_nr * 100 / warmup_kv_count_in_this_ps_);
    else
      LOG(INFO) << "All warmed KVs inserted";

    LOG(INFO) << "server " << server_id_ << " after check db";
  }

private:
  BaseKV* kv_;
  int server_id_;
  int warm_thread_count_;
  uint64_t warmup_kv_count_in_this_ps_;
  int64_t warm_capacity_;
  int value_size_;
};
