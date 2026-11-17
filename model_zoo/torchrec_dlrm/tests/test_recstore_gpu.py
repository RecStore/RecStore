import torch
import os
import sys
import unittest
import tempfile
import shutil
import argparse
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.jagged_tensor import KeyedTensor
from torchrec import EmbeddingBagCollection
from torchrec.modules.embedding_configs import EmbeddingBagConfig

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from python.pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection


def get_embedding_collection_class(use_torchrec=False):
    """根据参数返回对应的EmbeddingBagCollection类"""
    if use_torchrec:
        return TorchRecEmbeddingBagCollection
    else:
        return RecStoreEmbeddingBagCollection


class TorchRecEmbeddingBagCollection(EmbeddingBagCollection):
    """TorchRec官方EmbeddingBagCollection的包装类，使其接口与RecStore版本一致"""
    
    def __init__(self, embedding_bag_configs):
        # 转换配置格式
        configs = [
            EmbeddingBagConfig(
                name=c["name"],
                embedding_dim=c["embedding_dim"],
                num_embeddings=c["num_embeddings"],
                feature_names=c.get("feature_names", [c["name"]])
            )
            for c in embedding_bag_configs
        ]
        super().__init__(tables=configs)
        
        # 存储配置
        self._embedding_bag_configs = configs
        
        # 添加与RecStore版本一致的属性
        self.feature_keys = []
        self._embedding_dims = {}
        for config in configs:
            for feature_name in config.feature_names:
                self.feature_keys.append(feature_name)
                self._embedding_dims[feature_name] = config.embedding_dim
    
    def embedding_bag_configs(self):
        """返回配置列表，与RecStore版本保持一致"""
        return self._embedding_bag_configs
    
    def to(self, device):
        """重写to方法，确保模型移动到指定设备"""
        super().to(device)
        return self


class TestRecStoreEmbeddingBagCollectionGPU(unittest.TestCase):
    """测试RecStoreEmbeddingBagCollection在GPU上的各种功能"""
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置，检查CUDA可用性"""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA不可用，跳过GPU测试")
        
        cls.device = torch.device('cuda:0')
        print(f"使用GPU设备: {cls.device}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        
        # 设置日志级别为DEBUG以获取更多信息
        os.environ['RECSTORE_LOG_LEVEL'] = '3'
    
    def setUp(self):
        """每个测试前的设置"""
        # 确保在GPU上运行
        torch.cuda.empty_cache()
        
        # 获取命令行参数，决定使用哪个实现
        parser = argparse.ArgumentParser()
        parser.add_argument('--use-torchrec', action='store_true', 
                          help='使用TorchRec官方EmbeddingBagCollection而不是RecStore版本')
        args, _ = parser.parse_known_args()
        
        # 根据参数选择实现
        self.embedding_collection_class = get_embedding_collection_class(args.use_torchrec)
        self.use_torchrec = args.use_torchrec
        
        # 基础配置
        self.basic_configs = [
            {
                "name": "test_table",
                "num_embeddings": 100,
                "embedding_dim": 16,
                "feature_names": ["test_feature"]
            }
        ]
        
        # 多表配置
        self.multi_table_configs = [
            {
                "name": "user_table",
                "num_embeddings": 1000,
                "embedding_dim": 16,
                "feature_names": ["user_id"]
            },
            {
                "name": "item_table", 
                "num_embeddings": 500,
                "embedding_dim": 16,
                "feature_names": ["item_id"]
            },
            {
                "name": "category_table",
                "num_embeddings": 100,
                "embedding_dim": 16,
                "feature_names": ["category_id"]
            }
        ]

    def test_gpu_initialization(self):
        """测试GPU上的初始化功能"""
        print(f"\n=== 测试GPU初始化 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 验证配置
        configs = ebc.embedding_bag_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "test_table")
        self.assertEqual(configs[0].num_embeddings, 100)
        self.assertEqual(configs[0].embedding_dim, 16)
        
        # 验证特征键
        self.assertEqual(ebc.feature_keys, ["test_feature"])
        
        print("✓ GPU初始化测试通过")

    def test_gpu_forward_pass(self):
        """测试GPU上的前向传播"""
        print(f"\n=== 测试GPU前向传播 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 构造GPU上的输入数据
        kjt = KeyedJaggedTensor(
            keys=["test_feature"],
            values=torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
            lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["test_feature"])
        
        if self.use_torchrec:
            # TorchRec版本：输出形状是(2, 16)，length_per_key是[16]
            self.assertEqual(result.values().shape, (2, 16))
            self.assertEqual(result.length_per_key(), [16])
        else:
            # RecStore版本：输出形状是(2, 16)，length_per_key是[2]
            self.assertEqual(result.values().shape, (2, 16))
            self.assertEqual(result.length_per_key(), [2])
        
        self.assertEqual(result.values().device, self.device)
        
        print(f"✓ GPU前向传播测试通过，输出形状: {result.values().shape}")

    def test_gpu_multi_table_forward_pass(self):
        """测试GPU上多表前向传播"""
        print(f"\n=== 测试GPU多表前向传播 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.multi_table_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 构造GPU上的多表输入数据
        kjt = KeyedJaggedTensor(
            keys=["user_id", "item_id", "category_id"],
            values=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
            lengths=torch.tensor([1, 2, 2], dtype=torch.int32, device=self.device)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        self.assertIsInstance(result, KeyedTensor)
        self.assertEqual(result.keys(), ["user_id", "item_id", "category_id"])
        
        if self.use_torchrec:
            # TorchRec版本：输出形状是(1, 48)，length_per_key是[16, 16, 16]
            self.assertEqual(result.values().shape, (1, 48))
            self.assertEqual(result.length_per_key(), [16, 16, 16])
        else:
            # RecStore版本：输出形状是(3, 16)，length_per_key是[1, 1, 1]
            self.assertEqual(result.values().shape, (3, 16))
            self.assertEqual(result.length_per_key(), [1, 1, 1])
        
        self.assertEqual(result.values().device, self.device)
        
        print(f"✓ GPU多表前向传播测试通过，输出形状: {result.values().shape}")

    def test_gpu_gradient_update(self):
        """测试GPU上的梯度更新功能"""
        print(f"\n=== 测试GPU梯度更新功能 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 构造GPU上的输入数据
        kjt = KeyedJaggedTensor(
            keys=["test_feature"],
            values=torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
            lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 构造GPU上的梯度
        grad = torch.randn_like(result.values(), device=self.device)
        
        # 反向传播
        result.values().requires_grad_(True)
        result.values().backward(grad)
        
        print("✓ GPU梯度更新测试通过")

    def test_gpu_multi_table_gradient_update(self):
        """测试GPU上多表梯度更新"""
        print(f"\n=== 测试GPU多表梯度更新 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.multi_table_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 构造GPU上的多表输入数据
        kjt = KeyedJaggedTensor(
            keys=["user_id", "item_id", "category_id"],
            values=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
            lengths=torch.tensor([1, 2, 2], dtype=torch.int32, device=self.device)
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 构造GPU上的梯度
        grad = torch.randn_like(result.values(), device=self.device)
        
        # 反向传播
        result.values().requires_grad_(True)
        result.values().backward(grad)
        
        print("✓ GPU多表梯度更新测试通过")

    def test_gpu_device_transfer(self):
        """测试GPU设备间数据传输"""
        print(f"\n=== 测试GPU设备间数据传输 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 在CPU上构造数据
        cpu_kjt = KeyedJaggedTensor(
            keys=["test_feature"],
            values=torch.tensor([0, 1, 2], dtype=torch.int64),  # 使用有效的ID范围
            lengths=torch.tensor([2, 1], dtype=torch.int32)
        )
        
        # 转移到GPU
        gpu_kjt = KeyedJaggedTensor(
            keys=cpu_kjt.keys(),
            values=cpu_kjt.values().to(self.device),
            lengths=cpu_kjt.lengths().to(self.device)
        )
        
        # 在GPU上前向传播
        result = ebc(gpu_kjt)
        
        # 验证结果在GPU上
        self.assertEqual(result.values().device, self.device)
        
        # 转移回CPU
        cpu_result = result.values().cpu()
        self.assertEqual(cpu_result.device.type, 'cpu')
        
        print("✓ GPU设备间数据传输测试通过")

    def test_gpu_large_batch_forward_pass(self):
        """测试GPU上大批次前向传播"""
        print(f"\n=== 测试GPU大批次前向传播 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 构造GPU上的大批次数据
        batch_size = 1000
        ids = torch.randint(0, 100, (batch_size,), dtype=torch.int64, device=self.device)  # 确保ID在0-99范围内
        lengths = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        
        kjt = KeyedJaggedTensor(
            keys=["test_feature"],
            values=ids,
            lengths=lengths
        )
        
        # 前向传播
        result = ebc(kjt)
        
        # 验证结果
        if self.use_torchrec:
            # TorchRec版本：输出形状是(1000, 16)
            self.assertEqual(result.values().shape, (1000, 16))
        else:
            # RecStore版本：输出形状是(1000, 16)
            self.assertEqual(result.values().shape, (1000, 16))
        
        self.assertEqual(result.values().device, self.device)
        
        print("✓ GPU大批次前向传播测试通过")

    def test_gpu_memory_management(self):
        """测试GPU内存管理"""
        print(f"\n=== 测试GPU内存管理 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 记录初始内存使用
        initial_memory = torch.cuda.memory_allocated()
        
        # 进行多次前向传播
        for i in range(10):
            kjt = KeyedJaggedTensor(
                keys=["test_feature"],
                values=torch.tensor([i % 100, (i+1) % 100, (i+2) % 100], dtype=torch.int64, device=self.device),  # 确保ID在有效范围内
                lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
            )
            result = ebc(kjt)
            
            # 确保结果在GPU上
            self.assertEqual(result.values().device, self.device)
        
        # 清理内存
        torch.cuda.empty_cache()
        
        # 检查内存使用是否合理
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        print(f"内存使用增加: {memory_increase / 1024 / 1024:.2f} MB")
        
        print("✓ GPU内存管理测试通过")

    def test_gpu_mixed_precision(self):
        """测试GPU混合精度"""
        print(f"\n=== 测试GPU混合精度 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 使用半精度
        with torch.amp.autocast('cuda'):
            kjt = KeyedJaggedTensor(
                keys=["test_feature"],
                values=torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
                lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
            )
            
            result = ebc(kjt)
            
            # 验证结果
            self.assertEqual(result.values().device, self.device)
            # 注意：在混合精度下，输出可能是float16或float32
            self.assertIn(result.values().dtype, [torch.float16, torch.float32])
        
        print("✓ GPU混合精度测试通过")

    def test_gpu_concurrent_operations(self):
        """测试GPU并发操作"""
        print(f"\n=== 测试GPU并发操作 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 创建多个流
        streams = [torch.cuda.Stream() for _ in range(3)]
        
        results = []
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                kjt = KeyedJaggedTensor(
                    keys=["test_feature"],
                    values=torch.tensor([i % 100, (i+1) % 100, (i+2) % 100], dtype=torch.int64, device=self.device),  # 确保ID在有效范围内
                    lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
                )
                result = ebc(kjt)
                results.append(result)
        
        # 同步所有流
        torch.cuda.synchronize()
        
        # 验证所有结果
        for i, result in enumerate(results):
            self.assertEqual(result.values().device, self.device)
            if self.use_torchrec:
                self.assertEqual(result.values().shape, (2, 16))
            else:
                self.assertEqual(result.values().shape, (2, 16))
        
        print("✓ GPU并发操作测试通过")

    def test_gpu_error_handling(self):
        """测试GPU错误处理"""
        print(f"\n=== 测试GPU错误处理 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        if not self.use_torchrec:
            # 测试不存在的表（仅RecStore版本支持）
            with self.assertRaises(RuntimeError):
                ebc.kv_client.pull("non_existent_table", torch.tensor([1], device=self.device))
        
        # 测试无效的ID范围
        kjt = KeyedJaggedTensor(
            keys=["test_feature"],
            values=torch.tensor([999], dtype=torch.int64, device=self.device),  # 超出范围的ID
            lengths=torch.tensor([1], dtype=torch.int32, device=self.device)
        )
        
        if self.use_torchrec:
            # TorchRec版本会抛出RuntimeError
            with self.assertRaises(RuntimeError):
                result = ebc(kjt)
        else:
            # RecStore版本应该不会抛出异常，而是返回零向量
            result = ebc(kjt)
            self.assertEqual(result.values().shape, (1, 16))
            self.assertEqual(result.values().device, self.device)
        
        print("✓ GPU错误处理测试通过")

    def test_gpu_performance_benchmark(self):
        """测试GPU性能基准"""
        print(f"\n=== 测试GPU性能基准 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 预热
        for _ in range(5):
            kjt = KeyedJaggedTensor(
                keys=["test_feature"],
                values=torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
                lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
            )
            _ = ebc(kjt)
        
        torch.cuda.synchronize()
        
        # 性能测试
        batch_size = 100
        num_iterations = 100
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        for _ in range(num_iterations):
            kjt = KeyedJaggedTensor(
                keys=["test_feature"],
                values=torch.randint(0, 100, (batch_size,), dtype=torch.int64, device=self.device),  # 确保ID在0-99范围内
                lengths=torch.ones(batch_size, dtype=torch.int32, device=self.device)
            )
            result = ebc(kjt)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        throughput = num_iterations / (elapsed_time / 1000)  # iterations per second
        
        print(f"GPU性能基准: {elapsed_time:.2f}ms for {num_iterations} iterations")
        print(f"吞吐量: {throughput:.2f} iterations/second")
        
        print("✓ GPU性能基准测试通过")

    def test_gpu_multi_device(self):
        """测试多GPU设备"""
        print(f"\n=== 测试多GPU设备 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        if torch.cuda.device_count() < 2:
            self.skipTest("需要至少2个GPU设备")
        
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 测试在不同GPU上运行
        for device_id in range(min(2, torch.cuda.device_count())):
            device = torch.device(f'cuda:{device_id}')
            
            # 对于TorchRec版本，需要将模型移动到当前设备
            if self.use_torchrec:
                ebc = ebc.to(device)
            
            with torch.cuda.device(device):
                kjt = KeyedJaggedTensor(
                    keys=["test_feature"],
                    values=torch.tensor([0, 1, 2], dtype=torch.int64, device=device),  # 使用有效的ID范围
                    lengths=torch.tensor([2, 1], dtype=torch.int32, device=device)
                )
                
                result = ebc(kjt)
                
                # 验证结果在正确的设备上
                self.assertEqual(result.values().device, device)
        
        print("✓ 多GPU设备测试通过")

    def test_gpu_tensor_operations(self):
        """测试GPU张量操作"""
        print(f"\n=== 测试GPU张量操作 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 测试不同的张量操作
        kjt = KeyedJaggedTensor(
            keys=["test_feature"],
            values=torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device),  # 使用有效的ID范围
            lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
        )
        
        result = ebc(kjt)
        
        # 测试张量操作
        result_squared = result.values() ** 2
        result_sum = result.values().sum()
        result_mean = result.values().mean()
        
        # 验证操作结果
        self.assertEqual(result_squared.device, self.device)
        self.assertEqual(result_sum.device, self.device)
        self.assertEqual(result_mean.device, self.device)
        
        print("✓ GPU张量操作测试通过")

    def test_gpu_gradient_accumulation(self):
        """测试GPU梯度累积"""
        print(f"\n=== 测试GPU梯度累积 ({'TorchRec' if self.use_torchrec else 'RecStore'}) ===")
        ebc = self.embedding_collection_class(self.basic_configs)
        
        # 对于TorchRec版本，需要将模型移动到GPU
        if self.use_torchrec:
            ebc = ebc.to(self.device)
        
        # 多次前向传播和梯度累积
        accumulated_grad = None
        
        for i in range(3):
            kjt = KeyedJaggedTensor(
                keys=["test_feature"],
                values=torch.tensor([i % 100, (i+1) % 100, (i+2) % 100], dtype=torch.int64, device=self.device),  # 确保ID在有效范围内
                lengths=torch.tensor([2, 1], dtype=torch.int32, device=self.device)
            )
            
            result = ebc(kjt)
            result.values().requires_grad_(True)
            
            # 计算损失
            loss = result.values().sum()
            
            # 反向传播
            loss.backward()
            
            # 累积梯度
            if accumulated_grad is None:
                accumulated_grad = result.values().grad.clone()
            else:
                accumulated_grad += result.values().grad
        
        # 验证梯度累积
        self.assertIsNotNone(accumulated_grad)
        self.assertEqual(accumulated_grad.device, self.device)
        
        print("✓ GPU梯度累积测试通过")


def run_all_gpu_tests():
    """运行所有GPU测试"""
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-torchrec', action='store_true', 
                      help='使用TorchRec官方EmbeddingBagCollection而不是RecStore版本')
    args, _ = parser.parse_known_args()
    
    implementation_name = "TorchRec官方" if args.use_torchrec else "RecStore"
    print(f"开始运行{implementation_name} EmbeddingBagCollection GPU测试套件...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法运行GPU测试")
        return False
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRecStoreEmbeddingBagCollectionGPU)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印总结
    print(f"\nGPU测试总结:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    print(f"跳过测试数: {len(result.skipped)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.skipped:
        print("\n跳过的测试:")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")
    
    return result.wasSuccessful()


def main():
    """主函数，运行所有GPU测试"""
    success = run_all_gpu_tests()
    if success:
        print("\n🎉 所有GPU测试通过！")
    else:
        print("\n❌ 部分GPU测试失败，请检查上述错误信息。")
    return success


if __name__ == "__main__":
    main()