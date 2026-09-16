# Paged cache：小算子组合与原生 scatter 实测

2026-09-16，测试提交 `7a9d908af4858309add3b1777dbeda97f6d3568d`。

**原生 `npu_scatter_nd_update_` 在本轮支持的布局上有明显收益，
但输出 cache 的 feature stride=2 会报错，目前不能无条件完整替换原实现。**

## 测试范围与结果

比较 [独立脚本](../../compare_paged_cache_scatter_nd.py) 中的两条完整路径：
PR #16252 的 `small_ops` 与 `scatter_nd`。
后者的计时包含 slot 转换、valid mask、二维坐标生成、dtype 转换和原生 scatter。
没有运行整网，也没有修改远端服务的算子实现。

- 共 39 个 case：35 个双方通过，3 个旧组合特殊数值不满足逐字节标准，1 个原生接口报错。
- 原生版本：38 个可执行 case 全部与独立 CPU 参考逐字节一致；另 1 个报错，不能算通过。
- 37 个可执行非空 case 都完成了图捕获后的 4 次输入变化重放，共 148 次；空输入不捕获图。
- 33 个有效性能 case，eager 和 graph 各测 15 轮交错 A/B。两个模式均无耗时退化，
  每个模式的 495/495 对样本都是原生路径更快。
- graph 加速比为 **2.00×～4.42×**，耗时降低 **50.00%～77.37%**。
  eager 加速比为 **1.83×～4.07×**，耗时降低 **45.44%～75.43%**。

性能结论只覆盖通过精度检查并实际计时的 33 个 case。
4 个特殊数值 case 和空输入只做精度检查；接口报错 case 没有性能数字。

## 代表 case 的完整调用耗时

下表为 graph Event 中位数，单位 μs。默认 BF16、D=512、H=1、block=640，
页间 padding 为 47616 个元素。`mixed` 包含 `-1`、capacity 和整数极值。

| Case | 小算子组合 | 原生 scatter 完整路径 | 耗时变化 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| T=1，全有效 | 75.340 | 18.063 | -76.02% | 4.17× |
| T=64，mixed | 123.381 | 41.690 | -66.21% | 2.96× |
| T=2048，mixed | 216.666 | 81.936 | -62.18% | 2.64× |
| T=4096，全有效 | 235.544 | 111.582 | -52.63% | 2.11× |
| T=4096，mixed | 292.549 | 97.988 | -66.51% | 2.99× |
| T=4096，FP32→BF16，全有效 | 250.304 | 114.147 | -54.40% | 2.19× |
| T=4096，projection 行间隔额外 512，全有效 | 264.181 | 114.522 | -56.65% | 2.31× |
| T=4096，slot stride=2，全有效 | 256.847 | 128.434 | -50.00% | 2.00× |

例如 T=4096、mixed 在 eager 下是 **301.228→110.375 μs，降低 63.36%**。
eager 的 Event 间隔可能包含 CPU 提交间隙，应优先用 graph 结果评估固定形状调用。
所有 case 的两种模式都保留在 [summary.csv](summary.csv)，原始样本在 [results.json](results.json)。

## 精度变化

整个 backing allocation 都纳入检查，包括 cache、页间 padding 和首尾保护区；
同时检查 source、slots 未被修改。原生可执行 case 的差异字节数和最大有限数绝对误差均为 0。

旧组合的 3 个失败 case 是特殊编码发生变化，不是本轮观察到的有限数值误差。
以下为首次 eager 写入的定位结果，单位为不同的 16 位元素个数：

| Case | 旧组合差异元素 | 原因 | 原生差异元素 |
| --- | ---: | --- | ---: |
| BF16，不写 slot 0 | 128 | 128 个 NaN 编码被规范化 | 0 |
| BF16，写 slot 0 | 129 | 128 个 NaN 编码变化，1 个负零变正零 | 0 |
| FP16，写 slot 0 | 3 | 2 个 NaN 编码变化，1 个负零变正零 | 0 |
| FP16，不写 slot 0 | 0 | 无差异 | 0 |

具体例子：BF16 的 `0x7fc1`、`0x7f81` 被旧组合改成 `0x7fff`，
负零 `0x8000` 被改成正零 `0x0000`。原生路径保留原编码。
旧组合会对 values 执行选择和归约；原生路径直接写 updates，不经过 slot 0 的求和恢复逻辑。
上述逐字节差异不能仅靠 `atol=0` 的有限数值比较识别。

图重放改变 values 和 slots 后，BF16 旧组合的最大差异为 379 字节，
FP16 写 slot 0 的最大差异为 5 字节；对应原生结果均为 0 字节。
详见 [diagnostics.json](diagnostics.json) 和逐 case 数据。

## 不能完整替换的布局

`t4096_feature_stride2` 中，旧组合 eager 精度通过，原生 eager 调用直接失败：

```text
aclnnScatterNdUpdate: 561103 / AclNN_Parameter_Error(EZ1001)
Output tensor should not from workspace.
```

原始 case 随后终止，因此没有把该 case 的 graph 或性能记为通过。
进一步用 T=64 分离输入和输出布局，得到：

| cache 最后一维 stride | values 最后一维 stride | 结果 |
| ---: | ---: | --- |
| 1 | 2 | 原生写入精确，差异 0 字节 |
| 2 | 1 | 同样报错 |
| 2 | 2 | 同样报错 |
| 2 | 2，先对 values 调用 contiguous | 同样报错 |

这将问题定位到当前环境对输出 cache 布局的支持，不能靠整理 values 解决。
页间有 padding、storage offset 非零但 feature stride=1 的布局，本轮可以正确写入。
实际接入需要保留对不支持输出布局的处理，或先明确调用方不会产生这类布局。
本次只验证直接替换，没有为了通过用例加入隐式回退。

## 设备任务证据

T=64 和 T=4096 的 padding+mixed case 分别采集 32 次完整写入。
`kernel_details.csv` 可见条目由 **640 降到 384，即每次 20→12**，见 [kernels.csv](kernels.csv)。

旧组合可见 `SelectV2`、`ReduceSum`、`ReduceAny`，以及 `IndexCheck`、
`AsStrided`、`IndexPutV2`、`ViewCopy`。原生路径中后半段变为一次 `ScatterNdUpdate`，
仍有 valid 判断、除法/取余、Stack 和 Where 等坐标准备操作。
因此整条原生路径还不是一个 kernel；收益来自减少数据选择、归约和索引写入周边工作。

这些是 profiler 中可见的 kernel 条目，不包含全部运行时 memcpy。
当前 profiler 的 ACL→NPU flow 关联解析失败，未用它推断完整 CPU/设备依赖链；
上面的性能数字来自独立、未开启 profiler 的 Event 测量。

## 环境与可复现性

- Ascend A3，`Ascend910_9382`，物理卡 7。
- PyTorch `2.10.0+cpu`、torch-npu `2.10.0.post4`；CANN 安装目录名 `cann-9.1.0`。
- npu-smi/驱动显示版本 `26.1.1`。
- graph：每次重放 32 次完整写入，每个计时样本重放 8 次，15 轮交错 A/B，取中位数。
- 整个主测试期间，同组物理卡 6 的 120 次有效采样，AICore/AIVector 均为 0。
  设备上有其他驻留模型进程，本轮是共享环境的空闲计算窗口，不是独占测试。
- 同一批 cache 地址反复写入，数据来自合成输入，不代表整网或跨层工作集收益。
- 主测试最终退出码为 1，原因是 3 个旧组合严格精度失败和 1 个原生接口报错。

在已配置的 NPU 环境，进入 `benchmarks` 后运行：

```bash
python compare_paged_cache_scatter_nd.py --device 0 --suite full --output full_results
python compare_paged_cache_scatter_nd.py --device 0 --case t4096_feature_stride2 --accuracy-only --output layout_repro
python compare_paged_cache_scatter_nd.py --device 0 --case bits_bfloat16_zero1 --accuracy-only --output bits_repro
```

[manifest.json](manifest.json) 记录被测脚本和完整证据包的 SHA256。
发布的结果保留全部性能和精度数字；错误堆栈仅缩写掉服务器目录、PID 和冗长路径。
完整未缩写日志及 profiler 原始数据保留在本轮验证目录。
