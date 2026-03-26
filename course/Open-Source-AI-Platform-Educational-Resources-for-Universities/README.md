# 高校人工智能开源平台课程资源

## 概述

本项目为面向高校的 AI 系统软件课程资源集合，涵盖从大模型通识到高性能算子、编译器优化、分布式训练及性能评测的完整知识体系。课程基于 FlagOS 开源软件栈构建，配套中英文讲义、作业和教学视频。

## 课程结构

| 章节 | 主题 | 课时 | 说明 |
|------|------|------|------|
| 01 | [AI 大模型通识课](01-Introduction-to-Large-AI-Models/) | — | 入门模块，建立大模型整体认知 |
| 02 | [智算系统软硬件基础（全量汇总）](02-AI-Computing-Hardware-and-Software-Fundamentals/) | 01–48 | 课程总目录，包含全部讲义、作业和视频索引 |
| 03 | [AI 系统软件基础与异构计算](03-AI-System-Software-and-Heterogeneous-Computing/) | 01–09 | 子模块 I：硬件体系、异构编程、Triton 基础 |
| 04 | [高性能 AI 算子与算子工程](04-High-Performance-AI-Operators-and-Engineering/) | 10–18 | 子模块 II：Attention、融合、Profiling、算子库设计 |
| 05 | [AI 编译器原理与优化](05-AI-Compiler-Principles-and-Optimization/) | 19–27 | 子模块 III：IR、MLIR、Pass 优化、后端代码生成 |
| 06 | [分布式并行训练与通信](06-Distributed-Parallel-Training-and-Communication/) | 28–36 | 子模块 IV：数据/模型并行、通信优化、显存管理 |
| 07 | [性能评测与下一代内核生成](07-Performance-Benchmarking-and-Next-Gen-Kernel-Generation/) | 37–48 | 子模块 V：性能分析、自动调优、KernelGen、开源贡献 |

## 目录说明

每个子模块（03–07）包含以下子目录：

```
N-1-slides_zh/    # 中文讲义 + 作业 + 教学大纲
N-2-slides_en/    # 英文讲义 + 作业 + 教学大纲
N-3-videos_zh/    # 中文教学视频索引
```

Chapter 02 为课程总目录，汇总了全部 48 个课时的完整资源。

## 课程平台

- **FlagOS 官网**：https://flagos.io
- **在线实验平台**：https://flagos.io/OnlineLab
- **FlagGems GitHub**：https://github.com/FlagOpen/FlagGems

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，(c) 2026 FlagOS 社区。
