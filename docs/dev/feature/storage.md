# 扩展存储组件


## 新增 Index

```cpp
void Get(Key_t key, Value_t& pointer, unsigned tid) override;
Value_t Put(Key_t key, Value_t pointer, unsigned tid) override;
void BatchGet(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) override;
void BatchPut(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) override;
bool Delete(Key_t& key) override;
```

未命中返回 `kValueHandleNone`。注册：

```cpp
FACTORY_REGISTER(Index, MY_INDEX, MyIndex, const BaseKVConfig&);
```

在 `test_kvengine` 参数化中增加 `index.type` 组合。

## 新增 ValueStore

实现 `Alloc` / `Write` / `Read` / `Free` / `SlotCapacity`；DRAM 可实现 `DirectPtr`。注册后补充 `test_kvengine` 与配置样例。

## 新增 IOBackend

以 **page** 为单位；实现同步/异步读写、`AllocateBuffer`、完成轮询；能批量提交时覆盖 `BatchReadPages` / `BatchWritePages`。

```cpp
FACTORY_REGISTER(IOBackend, MY_IO, MyIoBackend, const BaseKVConfig&);
```

---

## 修改检查清单

| 改动 | 验证 |
|------|------|
| Index | `ctest -R test_kvengine` |
| ValueStore / DRAM allocator | `test_kvengine` |
| SSD allocator / IO | `test_io_backend` + SSD value 相关 `test_kvengine` |
| 配置校验行为 | `KVEngineCompositeConfigTest` |
| YCSB / 对比 | `.agents/skills/rs-benchmark-kvengine/SKILL.md` |
