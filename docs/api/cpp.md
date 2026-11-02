# CommonOp / KVClientOp

!!! note "待重写"

    按当前接口声明核对示例，并更新按算子拆分后的文件位置。

文件：`src/framework/op.h`、`src/framework/op.cc`

```cpp
class CommonOp {
 public:
  virtual void EmbRead(const RecTensor& keys, RecTensor& values) = 0;
  virtual void EmbWrite(const RecTensor& keys, const RecTensor& values) = 0;
  virtual void EmbUpdate(const std::string& table_name,
                         const RecTensor& keys,
                         const RecTensor& grads) = 0;
  virtual bool InitEmbeddingTable(const std::string& table_name,
                                  const EmbeddingTableConfig& config) = 0;
};
```

`KVClientOp` 是默认实现。它根据配置选择远程 PS client 或本地 HierKV runtime。
