# CommonOp / KVClientOp

文件：`src/framework/op.h`、`src/framework/op.cc`

```cpp
class CommonOp {
 public:
  CommonOp() = default;

  // 初始化：按 values 或 InitStrategy 批量初始化 embedding。
  virtual void EmbInit(const RecTensor& keys, const RecTensor& init_values) = 0;
  virtual void EmbInit(const RecTensor& keys, const InitStrategy& strategy) = 0;

  // 同步读写。
  virtual void EmbRead(const RecTensor& keys, RecTensor& values) = 0;
  virtual void EmbWrite(const RecTensor& keys, const RecTensor& values) = 0;

  virtual void EmbUpdate(const RecTensor& keys, const RecTensor& grads) = 0;
  virtual void EmbUpdate(const std::string& table_name,
                         const RecTensor& keys,
                         const RecTensor& grads) = 0;

  // 返回非负 tag，失败时抛异常。
  virtual int InitEmbeddingTable(const std::string& table_name,
                                 const EmbeddingTableConfig& config) = 0;

  // 异步预取：返回唯一 id，配合 IsPrefetchDone / WaitForPrefetch /
  // GetPretchResult 使用。
  virtual uint64_t EmbPrefetch(const RecTensor& keys,
                               const RecTensor& values) = 0;
  virtual bool IsPrefetchDone(uint64_t prefetch_id) = 0;
  virtual void WaitForPrefetch(uint64_t prefetch_id) = 0;
  virtual void GetPretchResult(uint64_t prefetch_id, RecTensor& values) = 0;
};
```

完整接口（`EmbExists` / `EmbDelete`、异步写、checkpoint 等）见 `src/framework/op.h`。

`KVClientOp` 是默认实现。它根据配置选择远程 PS client 或本地 HierKV runtime。
