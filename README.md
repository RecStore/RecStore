<p align="center">
  <img src="docs/assets/branding/recstore-full-medium-1430x380.png" alt="RecStore" width="640">
</p>

<!-- --8<-- [start:readme-content] -->

<p align="center">
  <em>High-performance distributed parameter storage for recommendation models.</em>
</p>

<p align="center">
  <a href="https://github.com/RecStore/RecStore/blob/main/README.md">English</a> | <a href="https://github.com/RecStore/RecStore/blob/main/README.zh-CN.md">简体中文</a>
</p>

RecStore is a high-performance parameter storage system designed to meet the growing demands of recommendation models in modern AI data centers. Unlike NLP and vision models, where computation dominates, recommendation models are bottlenecked by memory due to massive, trillion-scale sparse embedding parameters. RecStore addresses this challenge by abstracting heterogeneous, networked memory as a unified embedding pool. It provides key functionalities—parameter indexing, memory management, near-memory computation, and communication—as modular components within the system.

RecStore is:

- **Universal** - Easily integrated into existing DL frameworks via minimal OP implementations.
- **Efficient** - Optimized for the unique access patterns of recommendation models, leveraging GPU and NIC hardware acceleration.
- **Cost-effective** - Utilizes low-cost storage (e.g., SSDs, persistent memory) to expand memory capacity and reduce large model serving costs.




## Build RecStore

RecStore uses a Docker-based workflow: complete dependency setup and container initialization first, then build inside the container using the standard CMake flow.

See the complete build steps here: [Quickstart](https://recstore.github.io/RecStore/quickstart/)

## Documentation

Full documentation lives at [recstore.github.io/RecStore](https://recstore.github.io/RecStore/). Main sections:

- [Quickstart](https://recstore.github.io/RecStore/quickstart/) — environment setup and build
- [Configuration](https://recstore.github.io/RecStore/config/) — project configuration
- [Architecture](https://recstore.github.io/RecStore/architecture/overview/) — layered design
- [Performance](https://recstore.github.io/RecStore/dev/performance/) — performance analysis


## Citation

We would appreciate citations to the following papers:

<details markdown="1">
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

<!-- --8<-- [end:readme-content] -->
