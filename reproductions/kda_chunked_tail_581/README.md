# Chunked KDA 581-token tail reproduction

这是一个可独立转发的算子级复现包，包含此前保存的真实输入及输出。
不需要模型权重，不需要启动 vLLM 服务，也不需要应用 KDA 修复 PR。
随机生成同样 shape 的输入不保证复现相同误差。

## 完整配套下载

[下载独立复现 ZIP](kda581_reproduction.zip)。解压后直接按下列命令运行。
包内另附 `FP64_reference.npz`（FP64 输出与最终状态），可供算子团队直接数值比对；
`checksums.json` 记录所有配套文件的 SHA256。CPU 重算仍由脚本独立生成参考。

## 快速运行

仅在 CPU 上重算 FP64 参考，并审计包内已有的 NPU 输出：

```bash
cd reproductions/kda_chunked_tail_581
python -m pip install numpy
python reproduce_kda581.py --audit-only --output cpu-results
```

在 Ascend A5 环境重新调用原生算子，同时运行 581 和 padding 到 640 的对照：

```bash
python reproduce_kda581.py --backend fla --device npu:0 --repeats 5 --output npu-results
```

使用已经编译好的 vLLM Ascend 原生扩展时，也可运行：

```bash
python reproduce_kda581.py --backend vllm-ascend --device npu:0 --repeats 5 --output ascend-results
```

输出目录必须不存在。脚本先校验输入 SHA256，然后保存 FP64 参考、每次原生输出、
最终状态、软件版本和 `report.json`。`--audit-only` 只审计已保存输出，
不会执行 NPU；原生运行模式会将新执行的结果放在 `new_native_results` 中。
请比较新运行的数据，不要将 `archived_output_audit` 当作新硬件结果。

## 已保存的现象

误差定义为 `100 * RMS(actual - reference) / RMS(reference)`，不是最大逐元素误差。

| 对照 | 全部 581 token 输出误差 | 最后 5 token 输出误差 | 最终状态误差 |
| --- | ---: | ---: | ---: |
| 原始 581-token 调用 | 4.113025% | 57.316970% | 0.216458% |
| 640-token 中性 padding，取前 581 token | 0.318366% | 0.261912% | 0.216515% |

这些数值来自包内历史 NPU 输出的 CPU 复算，见 `archived_cpu_audit.json`。
它们证明该输入的输出尾块误差显著降低；最终状态误差基本没有变化。
具体 kernel 指令层根因尚未确认，也不代表整个模型的精度提升比例。
新版本算子如果已经修复，重新执行不一定出现原始误差。

## 输入和对照

| 参数 | Shape / 值 | Dtype |
| --- | --- | --- |
| Q、K、V、raw gate | `[1, 581, 8, 128]`，BSND | BF16 |
| beta | `[1, 581, 8]`，已经 sigmoid | FP32 |
| initial state | `[1, 8, 128, 128]`，非零，VK 布局 | FP32 |
| A_log | `[8]` | FP32 |
| dt_bias | `[1024]` | FP32 |
| chunk_size / scale | `64` / `128**-0.5` | — |
| cu_seqlens | `[0, 581]` | Host list |
| chunk_indices | `[0,0, 0,1, ..., 0,9]` | Host list |
| gate 参数 | `safe_gate=True, lower_bound=-5, use_gate_in_kernel=True` | — |
| state 参数 | `output_final_state=True, state_v_first=True` | — |

对照仅在序列末尾补 59 个 token：Q、K、V、processed beta 为 0，raw gate 为负无穷，
`cu_seqlens` 改为 `[0, 640]`。这些 token 的衰减系数为 1，状态增量为 0。
比较前 581 个真实 token 和最终状态。完整参数见 `case_manifest.json`。

BF16 数据以 uint16 原始位模式无损保存，其他输入为 FP32。
`kda581_case.npz` 的 SHA256 为：

```text
497d1b857b6fa814eb1bbadb77e852de60d02ec53c292c92b768f61afd54a6b7
```

## CPU 参考的来源与范围

`reproduce_kda581.py:reference()` 是自行实现的 NumPy FP64 逐 token 递推，
不是 CANN 或 FLA 官方 CPU golden。其公式来自
[Kimi Linear 论文 §3](https://arxiv.org/html/2510.26692v1#S3)，
并已与另一份独立 PyTorch FP64 参考交叉核对，relative RMS 约 `1.64e-16`。

采用 VK 状态布局，令 `g = -5 * sigmoid(exp(A_log) * (raw_gate + dt_bias))`：

```text
H_decay = H_previous * exp(g)[None, :]
residual = v - H_decay @ k
H = H_decay + beta * outer(residual, k)
o = (H @ q) / sqrt(128)
```

按 head 独立计算。Q/K 已在捕获前归一化，beta 已 sigmoid；
这个参考检验状态递推、输出与最终状态，不独立检验归一化和 beta 的前处理。
NPZ 含固定的算子边界张量和数值输出，不含模型权重、提示词或文本回答。

## 镜像来源确认

FLA `26.9.1+deva4a7958` 是下列官方 A5 镜像自带的版本。
已从该 manifest 对应的原始 image config 创建全新只读容器读取包版本，
并确认验证环境中的 FLA 原生库与 Python 绑定 SHA256 与原始镜像一致，
见 `stock_fla_origin.json`。

可在自己的环境查询实际安装版本：

```bash
python -c "from importlib.metadata import version; print(version('flash-linear-attention-npu-a5'))"
```

## 历史环境

- 官方镜像：`quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler`
- ARM64 manifest：`sha256:555ffa772734e39e4277d0c243d7dc3600e9f6ed3ac2c03eae4b3ad331661875`
- CANN：`9.2.0`；FLA：`26.9.1+deva4a7958`
- 被测 release 基线：`7672a5b509467288a12d7a2cee3451dd939d8b9c`
- 原修复提交：`538dbee1bfdb1f678e350063b53025897d1d4cd7`
- 修复 PR：[main #19112](https://github.com/vllm-project/vllm-ascend/pull/19112)、
  [release #19041](https://github.com/vllm-project/vllm-ascend/pull/19041)

本分支只添加复现资料，不修改产品代码，不新增 nightly 测试。
打包后的脚本经过 CPU 审计；其重新执行 NPU 的结果应以使用者生成的报告为准。
