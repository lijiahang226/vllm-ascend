# Paged cache：原生 scatter 与 Triton 直接对比

2026-09-16，同一张 Ascend A3 卡、同一套输入、交错 A/B 实测。

**在本轮共同支持的布局上，优化后的 Triton 精度与原生相同，完整写 cache 路径更快。**
33 个性能 case 的中位数均为 Triton 更快：graph 加速 **1.97×～8.62×**，
耗时降低 **49.11%～88.40%**；eager 加速 **1.71×～2.55×**，耗时降低 **41.41%～60.79%**。
graph 上限 8.62× 来自全无效 slot；排除全无效 case 后上限为 8.21×。

## 比较对象与口径

- 原生：`torch_npu.npu_scatter_nd_update_`，加上 slot 转换、合法性判断、二维坐标生成、
  无效坐标处理和 updates 类型转换。所有准备工作都在计时内。
- Triton：上一轮已经优化和验证过的 writer，冻结源码并校验 SHA256；
  在一个 kernel 内完成 slot 检查、地址计算、数据读取、转换和写入。
- 这是两种完整写入方案的比较，不是单独测原生裸 kernel；没有将前处理挪到计时外。
  此轮也没有修改 kernel 来迎合测量。
- 仅测单算子，没有运行整网。此次数据直接配对采集，没有拿上一次运行的数字相除。

## 精度

| 实现 | eager | 捕获图后改变输入重放 | 差异字节数 / 最大有限数绝对误差 |
| --- | --- | --- | --- |
| Triton | 39/39 通过 | 38 个非空 case × 4 次，共 152 次通过 | 0 / 0 |
| 原生 | 38/39 通过，1 个布局报错 | 37 个非空 case × 4 次，共 148 次通过 | 可执行 case 全部 0 / 0 |

两者共同执行的 38 个 case，彼此之间及相对独立 CPU 参考均逐字节一致。
包含 BF16、FP16、FP32、FP32→BF16，非法 slot、页间 padding、非零 storage offset、
strided slots、projection 切片、双 head、不同 block 和空输入。
4 个特殊数值 case 包含 NaN payload、正负零、Inf 和 subnormal，双方全部通过。
检查整个 backing allocation，包括保护区和布局间隙，同时确认 source、slots 不被修改。
合法 slot 唯一；未验证重复合法目标的覆盖顺序。

空输入在原始 JSON 的 graph 栏下记录 4 次普通空操作，不进行图捕获，未计入上表重放数。
主套件在 stride=2 的原生 eager 错误处终止该 case；Triton 的该布局图精度由
[独立补测](triton_stride.json) 完成，没有将未执行项记为通过。

## 与 GLM5.3 Flash 实际 shape 的对应关系

这轮是按已知参数构造的单算子测试，未抓取模型逐次调用的张量。
对照已保存的权重检查记录、集成源码和 2026-09-12 成功启动日志：

| 项目 | 已核对的 Flash 配置 / 调用 | 本轮覆盖 |
| --- | --- | --- |
| 写入维度 | `kv_lora_rank=512`，`qk_rope_head_dim=0` | D=512 对齐；这里是压缩 KV 的维度 |
| cache head | cache spec 的 `num_kv_heads=1` | 主场景 H=1 对齐；H=2 是泛化检查 |
| dtype | BF16 激活与 KV cache | BF16 对齐；其余 dtype 是扩展检查 |
| page block | 此前部署混合 cache 对齐后为 640 | 主场景 block=640；384 是边界检查 |
| values | 调用侧先对 `[T,512]` 做 RMSNorm，再交给 cache writer | 主测试为等价 `[T,1,512]`；不计上游 RMSNorm |
| 调度 T | 此前 MTP=3，decode 图档位为 4/8/16/32/64，prefill 每步预算 2048 | 本轮 T=1/64/256/2048/4096，缺少 4/8/16/32 |
| page stride / offset | 原整网未保存逐次 stride 和 storage offset | padding=0/47616、offset 是布局对照，不能声称等于实参 |
| 页数 / slot 分布 | 随 cache 容量和调度而变 | 合成小页池和随机唯一合法 slot，不是运行时回放 |

因此 **核心 dim、head、dtype 和此前部署 block 对齐，完整运行时 shape/stride 尚未逐项对齐**。
T=4096 超过此前每步 2048 的 token 预算，是压力检查；feature stride=2 也是泛化布局，
没有证据表明本轮 Flash 的该写入路径会产生它。
这里的倍率不能当作覆盖全部 Flash decode 档位或真实 slot 分布的性能保证。
本次追加读取远端最新配置时 SSH 两次超时，上述对应关系依据保留的配置与成功运行证据，
不宣称刚刚验证过远端权重文件未改变。

## Graph 性能

下面是 NPU Event 中位数。默认 BF16、H=1、D=512、block=640，
页间 padding=47616 个元素；mixed 混合合法 slot、`-1`、capacity 和整数极值。
加速比统一为 **原生耗时 / Triton 耗时**，降低比例为 `1 - Triton/原生`。

| Case | 原生完整路径 μs | Triton μs | Triton 加速比 | Triton 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| T=1，全有效 | 17.974 | 9.146 | 1.97× | 49.11% |
| T=64，mixed | 41.742 | 9.536 | 4.38× | 77.16% |
| T=2048，mixed | 78.647 | 9.576 | 8.21× | 87.82% |
| T=4096，全有效 | 111.423 | 24.316 | 4.58× | 78.18% |
| T=4096，mixed | 100.569 | 14.065 | 7.15× | 86.01% |
| T=4096，FP32→BF16，全有效 | 113.096 | 26.682 | 4.24× | 76.41% |
| T=4096，projection 行间隔额外 512 | 115.358 | 24.093 | 4.79× | 79.11% |
| T=4096，slot stride=2 | 127.619 | 24.119 | 5.29× | 81.10% |
| T=4096，全无效（不写 cache） | 83.410 | 9.673 | 8.62× | 88.40% |

33 个 case × 15 轮，共 **495/495 对 graph 样本为 Triton 更快**。
全无效 case 只检查 slot，不实际写 cache，不能将其作为有效写入的带宽收益。

## Eager 性能与波动

| Case | 原生完整路径 μs | Triton μs | Triton 加速比 | Triton 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| T=1，全有效 | 94.263 | 48.885 | 1.93× | 48.14% |
| T=64，mixed | 97.548 | 49.140 | 1.99× | 49.62% |
| T=2048，mixed | 94.323 | 51.235 | 1.84× | 45.68% |
| T=4096，全有效 | 121.060 | 51.215 | 2.36× | 57.69% |
| T=4096，mixed | 110.325 | 51.858 | 2.13× | 53.00% |
| T=4096，FP32→BF16，全有效 | 124.113 | 53.817 | 2.31× | 56.64% |
| T=4096，projection 行间隔额外 512 | 123.945 | 52.150 | 2.38× | 57.92% |
| T=4096，slot stride=2 | 131.825 | 51.682 | 2.55× | 60.79% |
| T=4096，全无效（不写 cache） | 109.783 | 49.513 | 2.22× | 54.90% |

eager 为 **494/495 对样本 Triton 更快**。唯一反向样本为 `t64_two_heads` 第 10 对：
Triton **295.255 μs**，原生 **106.775 μs**；没有删除这个样本。
该 case 的中位数仍是 Triton **50.698 μs**、原生 **107.425 μs**，Triton 快 **2.12×**。
本次没有建立这个孤立高耗时样本的具体原因，不能将其直接归因为 kernel 或其他进程。

eager Event 区间包含可能的 CPU 提交空隙，不能等同于裸 kernel 执行时间。
graph 减少了逐调用 Python 提交的影响；两种模式均保留同步墙钟时间和全部样本。

## 非连续特征布局的限制

`t4096_feature_stride2`：cache 和 values 的最后一维 stride 均为 2。

- 原生失败：`aclnnScatterNdUpdate: 561103 / EZ1001: Output tensor should not from workspace.`
  这是接口错误，不是数值误差；无可比较性能值。
- Triton eager 和 4 次图重放均逐字节正确，但 eager **2444.103 μs**、graph **2433.950 μs**。
  这个布局能正确执行，但性能很差，不能因“支持 stride”就认为所有布局都高效。
- 作为布局敏感性的参考，T=4096、同 dtype/slot/padding 的 feature stride=1 case，
  Triton graph 为 **24.316 μs**。两者存储跨度不同，不能把这个比值当成原生与 Triton 的加速比。

源码在 feature stride=1 时使用连续向量读写；stride=2 不走行分组优化。
离散读写及后端生成方式是后续定位方向，此次没有对该慢布局做底层指令分析，
不宣称已经证明其具体瓶颈。

## 为什么主要场景中 Triton 更快

T=64/4096 的 padding+mixed case，各采集 32 次完整写入：

| 实现 | profiler 可见 kernel 总条目 | 每次写入条目 |
| --- | ---: | ---: |
| 原生完整路径 | 384 | 12 |
| Triton | 32 | 1 |

原生路径仍有 GreaterEqual、Less、LogicalAnd、Fill、Select、FloorDiv、FloorMod、Stack，
最后才调用 ScatterNdUpdate。Triton 在单个 kernel 内完成相同语义，避免多个独立算子
之间的启动、调度和中间张量读写；无效 slot 可以直接跳过源数据读取和 cache 写入。
连续 D=512 场景还使用连续向量访问，较大 T 时按每组 8 行分配工作。
这些源码差异和任务数量支持收益方向；本次没有将收益精确拆分成启动开销与访存开销各占多少。

[kernels.csv](kernels.csv) 保留具体条目名称。这里统计的是 profiler 可见 kernel，
不包含全部运行时 memcpy；ACL→NPU flow 关联解析失败，未据此推断完整 CPU/设备依赖链。
性能表来自未开启 profiler 的独立测量。

## 环境、复现与结论边界

- Ascend A3，`Ascend910_9382`，物理卡 7；PyTorch `2.10.0+cpu`、torch-npu `2.10.0.post4`。
- npu-smi/驱动显示 `26.1.1`；沿用相同容器，CANN 安装目录名 `cann-9.1.0`。
- 每个 case 独立进程；预热后交错 15 轮 A/B，每个样本 8 次调用；
  graph 每次重放包含 32 次完整写入，按总调用数归一化。
- 主测试期间，同组物理卡 6 的 **193 次有效采样**，AICore/AIVector 均为 0。
  卡上仍有其他驻留模型进程，是共享环境的空闲计算窗口，不是独占保证。
- 同一批 cache 地址反复写入，合成输入；不代表跨层工作集、整网吞吐或任意形状的收益。
- 主套件退出码 1 来自原生不支持的布局；补测退出码 0。没有隐藏失败。
- 新增 Triton 基线后，默认 small_ops 模式另跑 T=64/padding/mixed，精度通过。
  本次默认路径 graph 为小算子 119.177 μs、原生 41.950 μs，只作兼容性回归记录。

进入已配置 NPU 环境的 `benchmarks` 目录运行：

```bash
python compare_paged_cache_scatter_nd.py --device 0 --baseline triton --suite full --output triton_native_results
python check_paged_cache_triton_stride.py
```

第二条命令单独验证原生不支持布局的 Triton eager/graph 精度，并输出 `triton_stride.json`。
先通过设备可见性设置选择空闲卡；补测脚本使用逻辑卡 0。
基准默认仍是 `small_ops`，需要明确传 `--baseline triton`。
原始 CSV 的 `speedup` 字段仍是 `triton/scatter_nd`，**本报告加速比取其倒数**。

[全部 case 汇总](summary.csv)、[完整样本与精度](results.json)、[汇总元数据](summary.json)
及 [源码/证据 SHA256](manifest.json) 一并保留。
发布时仅缩写了错误中的服务器路径、PID 和堆栈；精度和计时数字保持原样。
完整未缩写证据包和 profiler 原始目录保留在本轮远端验证目录。

本次只增加可重复的比较入口和冻结参考源码，不改变线上模型的算子选择。
在本轮主要连续特征布局中，应优先保留优化后的 Triton；原生版本易于调用，
但完整路径性能更低，并受输出布局支持范围限制。

本地 Ruff、格式、codespell、typos、Markdown 与长函数检查通过，
冻结脚本和 kernel 与设备执行文件逐字节一致，补测脚本的可执行 AST 一致。
全仓 `format.sh ci` 已执行，Windows 上 Gitleaks/logger 的 `/bin/bash` 和 shellcheck
依赖不可用，未宣称全仓 CI 通过。
