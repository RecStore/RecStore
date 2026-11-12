<p align="center">
  <img src="docs/assets/branding/recstore-full-medium-1430x380.png" alt="RecStore" width="640">
</p>

<p align="center">
  <em>面向推荐模型的高性能分布式参数存储系统。</em>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

RecStore 是一款高性能参数存储系统，旨在满足现代 AI 数据中心中推荐模型日益增长的需求。与以计算为主导的 NLP 和视觉模型不同，推荐模型因规模达万亿级的稀疏 embedding 参数而受制于内存。RecStore 通过将异构的、网络化的内存抽象为统一的 embedding 池来应对这一挑战，并将参数索引、内存管理、近存计算与通信等关键功能以模块化组件的形式提供于系统之内。

RecStore 具有如下特性：
- **Universal（通用）**：通过最少量算子实现即可便捷集成至现有深度学习框架。
- **Efficient（高效）**：针对推荐模型特有的访问模式进行优化，充分利用 GPU 与网卡硬件加速。
- **Cost-effective（低成本）**：利用 SSD、持久内存等低成本存储介质扩展内存容量，降低大模型服务成本。

## 构建 RecStore

RecStore 采用基于 Docker 的工作流：先完成依赖安装与容器初始化，再于容器内按标准 CMake 流程进行构建。

完整构建步骤见：[Quickstart](https://recstore.github.io/RecStore/quickstart/)

## 文档

完整文档位于 [recstore.github.io/RecStore](https://recstore.github.io/RecStore/)。主要章节如下：

- [Quickstart](https://recstore.github.io/RecStore/quickstart/) —— 环境搭建与构建
- [Configuration](https://recstore.github.io/RecStore/config/) —— 项目配置
- [Architecture](https://recstore.github.io/RecStore/architecture/overview/) —— 分层设计
- [Performance](https://recstore.github.io/RecStore/dev/performance/) —— 性能分析

## 引用

欢迎引用以下论文：

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{xie2025frugal,
	title={Frugal: Efficient and Economic Embedding Model Training with Commodity GPUs},
	author={Xie, Minhui and Zeng, Shaoxun and Guo, Hao and Gao, Shiwei and Lu, Youyou},
	booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1},
	pages={509--523},
	year={2025}
}

@inproceedings{fan2024maxembed,
	title={MaxEmbed: Maximizing SSD Bandwidth Utilization for Huge Embedding Models Serving},
	author={Fan, Ruwen and Xie, Minhui and Jiang, Haodi and Lu, Youyou},
	booktitle={The 29th Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'24)},
	year={2024}
}

@article{xie2023petps,
	title={PetPS: supporting huge embedding models with persistent memory},
	author={Xie, Minhui and Lu, Youyou and Wang, Qing and Feng, Yangyang and Liu, Jiaqiang and Ren, Kai and Shu, Jiwu},
	journal={Proceedings of the VLDB Endowment},
	volume={16},
	number={5},
	pages={1013--1022},
	year={2023},
	publisher={VLDB Endowment}
}

@inproceedings{xie2022fleche,
	title={Fleche: an efficient gpu embedding cache for personalized recommendations},
	author={Xie, Minhui and Lu, Youyou and Lin, Jiazhen and Wang, Qing and Gao, Jian and Ren, Kai and Shu, Jiwu},
	booktitle={Proceedings of the Seventeenth European Conference on Computer Systems},
	pages={402--416},
	year={2022}
}

@inproceedings{xie2020kraken,
	title={Kraken: memory-efficient continual learning for large-scale real-time recommendations},
	author={Xie, Minhui and Ren, Kai and Lu, Youyou and Yang, Guangxu and Xu, Qingxing and Wu, Bihai and Lin, Jiazhen and Ao, Hongbo and Xu, Wanhong and Shu, Jiwu},
	booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis},
	pages={1--17},
	year={2020},
	organization={IEEE}
}
```

</details>
