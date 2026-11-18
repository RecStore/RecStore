#!/usr/bin/env python3
"""
分布式DLRM训练示例
使用TorchRec进行大规模推荐系统模型的分布式训练

这个示例展示了如何使用TorchRec和RecStore进行分布式DLRM训练，
包括模型定义、数据加载、分布式设置和训练循环。
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim import KeyedOptimizerWrapper
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# 添加项目路径（相对本文件，避免绑定到某台机器的 checkout 位置）
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO_ROOT / "src/executable/dlrm"))
sys.path.append(str(_REPO_ROOT / "model_zoo/torchrec_dlrm"))

from dlrm_data_pytorch import CriteoDataset, collate_wrapper
from dlrm import DLRMTrain, InteractionType


class DistributedDLRMTrainer:
    """分布式DLRM训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format=f"[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def setup_distributed(self):
        """初始化分布式训练环境"""
        if self.world_size > 1:
            # 初始化进程组
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank
            )

            # 设置CUDA设备
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")

        self.logger.info(f"分布式训练设置完成: world_size={self.world_size}, rank={self.rank}")

    def create_embedding_configs(self) -> List[EmbeddingBagConfig]:
        """创建嵌入表配置"""
        # Criteo数据集的特征配置
        # 26个类别特征，每个特征的词汇表大小不同
        categorical_feature_sizes = [
            1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572
        ]

        embedding_configs = []
        for i, vocab_size in enumerate(categorical_feature_sizes):
            config = EmbeddingBagConfig(
                name=f"cat_{i}",
                embedding_dim=self.args.embedding_dim,
                num_embeddings=vocab_size,
                feature_names=[f"cat_{i}"],
            )
            embedding_configs.append(config)

        return embedding_configs

    def create_model(self) -> nn.Module:
        """创建DLRM模型"""
        embedding_configs = self.create_embedding_configs()

        # 创建嵌入表集合
        ebc = EmbeddingBagCollection(
            tables=embedding_configs,
            device=self.device,
        )

        # 创建DLRM模型
        model = DLRMTrain(
            embedding_bag_collection=ebc,
            dense_in_features=self.args.dense_features,
            dense_arch_layer_sizes=self.args.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.args.over_arch_layer_sizes,
            dense_device=self.device,
            interaction_type=InteractionType.ORIGINAL,
        )

        return model

    def setup_distributed_model(self, model: nn.Module) -> DistributedModelParallel:
        """设置分布式模型"""
        if self.world_size == 1:
            return model

        # 创建拓扑结构
        topology = Topology(
            world_size=self.world_size,
            compute_device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 创建分片规划器
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=self.args.batch_size,
        )

        # 规划分片策略
        plan = planner.collective_plan(
            module=model,
            sharders=[
                # 嵌入表使用表级分片
                {"sharding_type": ShardingType.TABLE_WISE.value},
                # 也可以使用行级分片: {"sharding_type": ShardingType.ROW_WISE.value}
            ],
            pg=dist.group.WORLD,
        )

        # 创建分布式模型
        distributed_model = DistributedModelParallel(
            module=model,
            plan=plan,
            device=self.device,
        )

        self.logger.info("分布式模型设置完成")
        return distributed_model

    def create_dataloader(self) -> DataLoader:
        """创建数据加载器"""
        # 创建Criteo数据集
        dataset = CriteoDataset(
            dataset=self.args.dataset_path,
            max_ind_range=self.args.max_ind_range,
            sub_sample_rate=self.args.sub_sample_rate,
            randomize="total" if self.args.randomize else "none",
            split="train",
            raw_path=self.args.raw_data_file,
            pro_data=self.args.processed_data_file,
        )

        # 分布式采样器
        sampler = None
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )

        # 数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=True,
            drop_last=True,
        )

        self.logger.info(f"数据加载器创建完成: batch_size={self.args.batch_size}")
        return dataloader

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """创建优化器"""
        # 为不同类型的参数设置不同的学习率
        dense_params = []
        sparse_params = []

        for name, param in model.named_parameters():
            if "embedding" in name:
                sparse_params.append(param)
            else:
                dense_params.append(param)

        # 使用AdaGrad优化器（推荐系统常用）
        optimizer = torch.optim.Adagrad([
            {"params": dense_params, "lr": self.args.learning_rate},
            {"params": sparse_params, "lr": self.args.sparse_learning_rate},
        ])

        # 如果是分布式模型，需要包装优化器
        if isinstance(model, DistributedModelParallel):
            optimizer = KeyedOptimizerWrapper(
                dict(model.named_parameters()),
                lambda params: torch.optim.Adagrad(params, lr=self.args.learning_rate)
            )

        return optimizer

    def train_step(self, model: nn.Module, batch: Tuple, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """单步训练"""
        dense_features, sparse_features, labels = batch

        # 移动数据到设备
        dense_features = dense_features.to(self.device)
        labels = labels.to(self.device)

        # 创建KeyedJaggedTensor用于稀疏特征
        if isinstance(sparse_features, KeyedJaggedTensor):
            sparse_features = sparse_features.to(self.device)
        else:
            # 如果不是KeyedJaggedTensor，需要转换
            sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=[f"cat_{i}" for i in range(len(sparse_features))],
                values=torch.cat(sparse_features).to(self.device),
                lengths=torch.tensor([len(feat) for feat in sparse_features]).to(self.device),
            )

        # 前向传播
        optimizer.zero_grad()
        logits = model(dense_features, sparse_features)
        loss = criterion(logits.squeeze(), labels.float())

        # 反向传播
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")

        # 创建模型
        model = self.create_model()
        model = self.setup_distributed_model(model)

        # 创建数据加载器
        dataloader = self.create_dataloader()

        # 创建优化器和损失函数
        optimizer = self.create_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()

        # 训练循环
        model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0.0
            num_batches = 0

            # 设置分布式采样器的epoch
            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                loss = self.train_step(model, batch, optimizer, criterion)
                total_loss += loss
                num_batches += 1

                # 打印训练进度
                if batch_idx % self.args.log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.args.epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Loss: {loss:.6f}"
                    )

                # 早停检查
                if batch_idx >= self.args.max_batches_per_epoch:
                    break

            # 计算平均损失
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}")

            # 保存检查点
            if self.rank == 0 and (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(model, optimizer, epoch, avg_loss)

        self.logger.info("训练完成!")

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        checkpoint_path = f"{self.args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def cleanup(self):
        """清理分布式环境"""
        if self.world_size > 1:
            dist.destroy_process_group()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分布式DLRM训练")

    # 模型参数
    parser.add_argument("--embedding-dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--dense-features", type=int, default=13, help="稠密特征数量")
    parser.add_argument("--dense-arch-layer-sizes", type=int, nargs="+", default=[512, 256, 128], help="底部MLP层大小")
    parser.add_argument("--over-arch-layer-sizes", type=int, nargs="+", default=[1024, 512, 256, 1], help="顶部MLP层大小")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=2048, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--sparse-learning-rate", type=float, default=0.01, help="稀疏参数学习率")

    # 数据参数
    parser.add_argument("--dataset-path", type=str, default="/data/criteo", help="数据集路径")
    parser.add_argument("--raw-data-file", type=str, default="train.txt", help="原始数据文件")
    parser.add_argument("--processed-data-file", type=str, default="kaggleAdDisplayChallenge_processed.npz", help="预处理数据文件")
    parser.add_argument("--max-ind-range", type=int, default=40000000, help="最大索引范围")
    parser.add_argument("--sub-sample-rate", type=float, default=1.0, help="子采样率")
    parser.add_argument("--randomize", action="store_true", help="是否随机化数据")

    # 系统参数
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--log-interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save-interval", type=int, default=1, help="保存检查点间隔")
    parser.add_argument("--max-batches-per-epoch", type=int, default=10000, help="每轮最大批次数")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="检查点保存目录")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建训练器
    trainer = DistributedDLRMTrainer(args)

    try:
        # 设置分布式环境
        trainer.setup_distributed()

        # 开始训练
        trainer.train()

    except Exception as e:
        trainer.logger.error(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 清理环境
        trainer.cleanup()


if __name__ == "__main__":
    main()