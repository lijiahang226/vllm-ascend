# 小算子组合与 scatter_nd_update 精度、性能对比

脚本：`compare_paged_cache_scatter_nd.py`。复制这一个 Python 文件到安装了
PyTorch、torch-npu 和对应 CANN 的机器即可运行，不依赖 vLLM、Triton 或以前的测试目录。
只测试写 cache 算子，不启动模型，不添加 nightly 用例。

另外支持 `--baseline triton`，直接对比上一轮验证过的 Triton writer 与原生完整路径。
该模式还需要同目录的 `paged_cache_triton_reference.py`，以及 vLLM、vllm-ascend、
Triton-Ascend 环境。脚本校验参考文件的固定 SHA256，防止误测其他 Triton 版本。

## 比较对象

| 名称 | 实际执行内容 |
| --- | --- |
| `small_ops` | PR #16252 的原版组合：valid/where、div/remainder、保存 slot 0、where/sum/any、高级索引写入、恢复 slot 0 |
| `scatter_nd` | slots 转 int64、valid/where、生成二维坐标、无效坐标转 `[-1,-1]`、updates 类型转换、`torch_npu.npu_scatter_nd_update_` |
| `triton`（可选基线） | 冻结的优化版 writer，在一个 kernel 内完成合法性判断、按 stride 寻址、转换与写入 |

原版来自提交 `517e0e5b8969e44077ad96d81e66f9bf76aa3f0f` 的
`vllm_ascend/attention/utils.py::scatter_paged_cache`，保留其 slot 0 归约逻辑，
并包含原调用侧的 slots/values 类型转换。原版存在精度问题时照实报告，不为跑分修改基线。

这里“融合算子”指用原生 scatter 替换写入部分；Python 生成坐标的操作仍然存在。
**计时包括两侧完整函数，不把索引生成、mask 或 dtype 转换移到计时外。**
未运行 profiler 前，不声称整段只有一个设备任务。

## 运行

先在已经初始化好 CANN 环境的终端进入脚本目录，选择一张空闲卡。
`--device` 是当前进程可见的逻辑卡号；如果设置了 `ASCEND_RT_VISIBLE_DEVICES`，
应按筛选后的编号传入。脚本不会清理其他进程或修改设备占用。
下面输出目录必须尚不存在，防止覆盖旧结果。

```bash
# 4 个快速场景：T=64/4096，连续/有页间 padding，有效/混合无效 slot。
python compare_paged_cache_scatter_nd.py --device 0 --suite smoke --output smoke_results

# 完整 39 个场景，每个默认测试 eager 和 NPUGraph。
python compare_paged_cache_scatter_nd.py --device 0 --suite full --output full_results

# 仅验证精度，不跑性能。
python compare_paged_cache_scatter_nd.py --device 0 --suite full --accuracy-only --output accuracy_results

# 指定一个典型 case，15 轮 A/B，每个 graph 包含 32 次完整调用。
python compare_paged_cache_scatter_nd.py --device 0 --case t4096_pad47616_mixed --modes graph --repeats 15 --unroll 32 --output one_case_results

# 列出全部场景及形状参数；此命令不需要安装 torch。
python compare_paged_cache_scatter_nd.py --suite full --list-cases

# 同一批输入上直接比较 Triton 与原生 scatter；默认原生路径仍包含完整前处理。
python compare_paged_cache_scatter_nd.py --device 0 --baseline triton --suite full --output triton_native_results
```

## 精度检查

两侧使用同一随机种子生成完全相同的 CPU 输入，各自分配独立设备 cache。
独立 CPU 参考按合法 slot 写入，其余位置保持初值。对比覆盖整个 backing allocation，
包括页间 padding、首尾保护区和非连续视图间隙，同时检查 values 和 slots 没有被修改。

- token 数：1、64、256、2048、4096；另有空输入。
- cache：默认 `[pages,640,1,512]`，页间 padding 为 0 或 47616 个元素；另测 block=384 和双 head。
- 类型：BF16、FP16、FP32，以及 FP32 输入写 BF16 cache；slot 为 int64 或 int32。
- 布局：非零 storage offset、slot stride=2、projection 切片、feature stride=2。
- slot：唯一合法目标、`-1`、capacity、整数最小/最大值、全无效；覆盖写入和不写入 slot 0。
- 特殊数值：16 位编码分布，明确放入 `+0/-0`、Inf、NaN payload 和 subnormal。
- 图模式：捕获后改变 values 和 slots，重放 4 次，每次重新从初始 cache 建立 CPU 参考。

主要标准是 **差异字节数为 0**。`max_abs_finite` 只辅助定位，不能替代逐字节标准：
例如负零变正零，绝对误差仍为 0，但字节已经不一致。
脚本同时给出两侧各自对 CPU 的误差，以及两侧直接对比的误差，避免“两边都写错但相互一致”。

原生版本是否忽略 `[-1,-1]`、是否正确原地写入非连续 cache，由当前机器实测决定。
报错或写错均为失败，不会自动换成其他实现。每个 case 使用独立子进程，
异常和超时记录到日志后继续其他 case。任一 case 失败时，脚本最终退出码为 1。

## 性能口径与结果

默认两侧交替执行 15 轮 A/B，保留所有样本，报告中位数。
graph 每次重放包含 32 次完整写 cache，按调用次数归一化为微秒；
eager 和 graph 均记录 NPU Event 和同步后的墙钟时间。
eager 的 Event 间隔可能包含主机提交不及时导致的设备空隙，不能解读成某个裸 kernel 的耗时。
这是同一 cache 地址反复写入的算子微基准，不代表整网或跨层大工作集的收益。

只有两侧该模式下的精度都通过，才生成可比较的性能数字。
空输入、特殊数值用例只验证精度；性能字段留空，不填 0。

输出文件：

- `summary.csv`：每 case/mode 的精度与性能汇总，可直接用表格软件查看。
- `results.json`：形状、torch/torch-npu 版本、设备型号、各轮误差、全部计时样本和脚本 SHA256。
- `<case>.json`、`<case>.log`：该 case 的结果与错误日志；子进程崩溃前已完成的检查会保留。

| 字段 | 含义 |
| --- | --- |
| `small_ops_bytes` / `scatter_nd_bytes` | 各自对 CPU 参考的最大差异字节数，跨重放取最大 |
| `pair_bytes` | 两种实现之间的最大差异字节数 |
| `small_ops_us` / `scatter_nd_us` | 完整调用的 Event 耗时中位数，单位 μs |
| `change_percent` | `(scatter_nd_us / small_ops_us - 1) × 100%`，负数表示耗时下降 |
| `speedup` | `small_ops_us / scatter_nd_us`，大于 1 表示原生路径更快 |
| `faster_pairs` / `pairs` | 原生路径更快的配对轮数 / 总轮数，用来观察波动 |

测量期间请避免其他任务使用同一张卡；脚本不保证设备独占。

使用 `--baseline triton` 时，CSV 中对应列为 `triton_bytes` 和 `triton_us`。
`change_percent` 表示原生相对 Triton 的耗时变化；`speedup=triton_us/scatter_nd_us`，
小于 1 表示原生更慢，倒数是 Triton 相对原生的加速比。`faster_pairs` 仍统计原生更快的轮数。

## 当前验证状态

**Triton 与原生的同卡直接 A/B 已完成，见 [Triton 对比报告](results/triton_native_20260916/README.md)。**
33 个性能 case 中，Triton graph 快 1.97×～8.62×；双方共同可执行的 38 个 case 均逐字节一致。
原生在 cache feature stride=2 时失败，Triton 该布局精度通过但性能较差。
维度和此前 Flash 部署的 block 已核对；合成 slot、padding 和 T 矩阵不等于真实调用回放。

**NPU 验证已完成，见 [完整实测报告](results/scatter_nd_20260916/README.md)。**
33 个有效性能 case 的图模式加速比为 2.00×～4.42×；
原生路径 38 个可执行 case 均逐字节通过，但输出 cache 的 feature stride=2 报错。
因此不能无条件完整替换。旧组合另有 3 个 NaN 编码/负零的严格精度失败 case。

2026-09-16：本地脚本语法、Ruff 和格式检查通过；确认 39 个完整 case、4 个快速 case，
命令行参数校验正常。AST 对比确认小算子算法与已保存的 PR 基线一致。
模拟本机缺少 torch 的真实启动失败，确认失败退出码、日志和空性能字段能正确保留。
这项失败处理检查不计为算子精度验证。
提交文件的 Ruff、格式、codespell、typos 和 Markdown 检查通过。
全仓 `format.sh ci` 已执行，但 Windows 环境缺少 `/bin/bash` 或 `shellcheck`，
导致 Gitleaks、logger 和 shellcheck 三个检查未能通过；不宣称全仓检查通过。
上述静态检查在无 torch/NPU 的本机完成；远端设备的测量数据、失败信息和共享环境限制见实测报告。
