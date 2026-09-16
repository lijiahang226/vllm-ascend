# 原生 scatter 直接调用与 Triton 对比

2026-09-16。原生侧计时/图捕获中只执行：

```python
torch_npu.npu_scatter_nd_update_(cache, indices, updates)
```

**这是原生入参已准备好的调用成本，不包含索引、mask、dtype 转换和数据整理。**
与 [此前包含前处理的结果](../triton_native_20260916/README.md) 是不同口径，不能混用倍率。
Triton 保留同一份冻结优化版 writer，在 kernel 内检查 slot、计算地址并写入。

## 实测结果

33 个共同可计时 case：

- eager：原生中位数更快 **33/33**，
  原生加速比范围 **1.20×～5.41×**；
  原生更快配对样本 **494/495**。
- graph：Triton 中位数更快 **32/33**，
  原生更快 **1/33**。
  Triton 加速比范围 **0.996×～2.30×**；
  原生更快配对样本 **14/495**。

这里原生加速比为 `Triton/原生`，Triton 加速比为 `原生/Triton`。
T=1 的图耗时约 9 μs，两者仅有很小差异，不能把略小的单次中位数视为稳定优势。
eager 包括主机提交间隙；裸原生接口与 Triton Python 启动路径的 CPU 成本也在 Event 区间内。
不能将 eager 的差异全部归因于设备内核。

默认 BF16、H=1、D=512、block=640、page padding=47616 个元素。
mixed 包含合法 slot、负数、capacity 和整数极值。下表为 15 轮交错 A/B 的 Event 中位数。

| Case | Eager Triton μs | Eager 原生 μs | Graph Triton μs | Graph 原生 μs |
| --- | ---: | ---: | ---: | ---: |
| T=1，全有效 | 46.725 | 9.370 | 8.913 | 8.929 |
| T=64，mixed | 47.318 | 15.290 | 9.579 | 14.713 |
| T=2048，mixed | 46.703 | 19.672 | 9.479 | 19.652 |
| T=4096，全有效 | 48.665 | 38.305 | 24.104 | 38.253 |
| T=4096，mixed | 48.125 | 32.340 | 14.323 | 30.232 |
| T=4096，FP32→BF16 | 49.103 | 38.255 | 26.332 | 37.300 |
| T=4096，projection 切片 | 48.795 | 38.107 | 24.116 | 37.233 |

## 精度与不支持的布局

主套件 39 个 case，双方共同执行的 38 个全部通过：相对独立 CPU 参考、彼此之间
均为 **0 字节差异，最大有限数绝对误差 0**。包含 BF16、FP16、FP32、FP32→BF16、
NaN payload、负零、Inf、subnormal、非法 slot、padding、不同 head/block、空输入。
检查整个 cache backing 和保护区，同时检查 slots、source 及原生 indices/updates 未被修改。

原生的 37 个非空 case 各完成 4 次捕获后改变输入重放，共 148 次。
每次重放前更新固定地址的 indices/updates；更新本身在图捕获之外。
空输入真实调用了原生接口且通过，但不进行空图捕获；JSON 的 graph 栏包含 4 次普通空操作。

另一个 case `t4096_feature_stride2` 的原生调用仍报错：
`aclnnScatterNdUpdate: 561103 / EZ1001: Output tensor should not from workspace.`
没有自动回退，也没有为该 case 生成性能值。该 case 的 Triton eager 通过；
本轮原生错误发生后未运行它的 graph，既有 Triton 图验证见前一份报告。

## 输入准备与计时边界

测试预先在 CPU 构造 `[T,2]` int64 的页号/页内位置；无效坐标为 `[-1,-1]`。
updates 预先转成 cache dtype，整理成 `[T,H,D]` 连续张量并拷贝到设备。
计时函数通过 `partial` 直接绑定原生 API，没有 Python 包装函数内的分支、where、div、stack 或 cast。
两侧先预热，graph 每次重放 32 次调用，每个计时样本重放 8 次。

这个假设对结果有影响：FP32→BF16 的原生输入已是 BF16，转换不计时；Triton 仍在 kernel 内转换。
projection/非连续 source 的原生 updates 也已经整理，Triton 读取原始 source view。
原始 source 有非零 offset；整理后的原生 updates offset=0。完整准备成本没有消失，
这里只按调用方已经具备入参的契约测量。

为检查整理 updates 是否影响本轮结论，额外直接比较原生 API 读取原始 source view 与整理后的 updates。
两边都预先提供同样语义的二维 indices，都只捕获原生算子；每个 case 的 4 次变化输入重放均通过。

| Case | 原生读取原始 view μs | 原生读取整理后 updates μs | 整理后耗时变化 |
| --- | ---: | ---: | ---: |
| `t64_pad47616_mixed` | 14.880 | 15.122 | +1.63% |
| `t4096_pad47616_mixed` | 33.722 | 29.950 | -11.18% |
| `t4096_projection_slice` | 45.854 | 36.589 | -20.21% |

这是单独配对采集的布局敏感性检查，不与主表跨轮相除。完整布局和样本见 [layout_check.json](layout_check.json)。

## 设备任务与 Flash 接入范围

T=64/4096 的 padding+mixed 场景各记录 32 次调用。
原生和 Triton 都是 **32 个可见 kernel 条目，即每次 1 个**。
具体 kernel 名称和条目数量见 [kernels.csv](kernels.csv)，汇总见 [summary.json](summary.json)。
profiler 不计全部运行时 memcpy，ACL→NPU flow 关联解析仍失败；性能数字来自未开启 profiler 的测量。

重新只读核对远端 GLM5.3 Flash 配置：`kv_lora_rank=512`、`qk_rope_head_dim=0`、dtype=BF16，
配置 SHA256 为 `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`，与此前部署一致。
cache H=1 和此前 block=640 有源码/启动证据。真实调用侧仍传入一维 slot，
只有上游提供了二维 indices 和正确 updates，才可以在 writer 处直接调用原生 API。
本次没有修改模型 metadata builder 或线上 writer。

T=4/8/16/32 档位、真实逐调用 stride、页池容量和 slot 分布仍未完整复现；
T=4096 是压力 case，超过此前部署的单步 2048 token 预算。不能将本表当作整网收益。

## 环境、复现和证据

- 物理 NPU8，Ascend910_9382；开始时卡8/9未显示计算进程。
- PyTorch 2.10.0+cpu、torch-npu 2.10.0.post4，驱动/npu-smi 26.1.1；沿用相同测试容器。
- 主流程期间同组物理卡9共 198 次有效采样，其中 0 次 AICore/AIVector 非零；未取得设备独占锁。
- 39 个 case 分进程执行；主套件退出码 1 来自上述原生布局错误，不计为全套通过。
- default full 模式另做 T64/padding/mixed 回归，结果见 [full_regression.json](full_regression.json)。

```bash
python compare_paged_cache_scatter_nd.py --device 0 --baseline triton --native-mode direct --suite full --output direct_results
```

默认 full 模式保留旧口径；使用 direct 必须显式指定。输出 CSV/JSON 标注 `native_mode`，
JSON 还标注 `native_preprocessing_timed=false` 和原生入参 shape/stride/dtype。
[完整样本](results.json)、[全部 case 表](summary.csv) 与 [SHA256 清单](manifest.json) 一并保存。
仅缩写了错误中的主机目录和 PID，精度与性能数字未改动。

Ruff、格式、codespell、typos、Markdown 和长函数检查通过。全仓 format.sh ci 已运行，
Windows 环境缺少 Gitleaks/logger 所需的 /bin/bash 和 shellcheck，未宣称全仓 CI 通过。
