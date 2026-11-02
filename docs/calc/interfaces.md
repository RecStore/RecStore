# 计算层接口

这一页记录计算层的边界接口：Python client、PyTorch op、C++ facade。模型代码通常只用 `RecStoreClient` 或 embedding 模块，不直接碰 `KVClientOp`。

## Python client

[RecStoreClient API](../api/python.md)

## PyTorch op

[PyTorch Operators](../api/operators.md)

## CommonOp / KVClientOp

[C++ 接口](../api/cpp.md)
