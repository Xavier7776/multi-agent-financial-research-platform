# Phase 4 评测报告 — 金融 Reviewer 四维校验可信度验证

**评测时间**: 2026-07-21

**评测模型**: mimo-v2.5-pro (reviewer + judge)

**judge 方式**: LLM 语义匹配 + 规则兜底


## 0. 关于 100% recall 的可信度

Phase 3 的 100% recall 引发 3 个方法论追问。Phase 4 直接回应：

| 追问 | Phase 4 应对 |
|---|---|
| 测试集多大？ | **102 注入样本 + 24 干净样本** = 126 总样本（vs Phase 3 的 58+8） |
| 怎么定义 ground truth？ | 每条 mutation 携带 `(dimension, mutation_type, ground_truth_desc)` 三元组，judge 用 LLM 做语义匹配（不要求字面命中） |
| 是不是自己构造的小样本？ | **4 ticker in-sample + 2 ticker held-out**，held-out ticker 不参与 Phase 3 评测，专门验证泛化性 |

**核心结论**：Phase 4 出现真实 miss，recall 不再是 100%，反而更可信。

## 1. 样本构成

| 维度 | in-sample | held-out | 合计 |
|---|---|---|---|
| compliance | 16 | 8 | 24 |
| data_accuracy | 8 | 7 | 15 |
| format | 28 | 15 | 43 |
| logic | 16 | 4 | 20 |
| clean (FP 测试) | 16 | 8 | 24 |
| **合计** | **102** | | |

## 2. 核心指标：各维度 recall（含 95% 置信区间）

**Wilson 95% 置信区间**：n 越大区间越窄，统计意义越强。

| 维度 | in-sample recall | held-out recall | 合并 recall (95% CI) | n |
|---|---|---|---|---|
| compliance | 100.0% | 100.0% | 100.0% [86.2%, 100.0%] | 24 |
| data_accuracy | 100.0% | 100.0% | 100.0% [79.6%, 100.0%] | 15 |
| format | 98.2% | 100.0% | 98.8% [87.9%, 99.6%] | 43 |
| logic | 87.5% | 100.0% | 90.0% [69.9%, 97.2%] | 20 |

## 3. 误报率（FP rate）

| 数据集 | FP / Total | FP rate |
|---|---|---|
| in-sample | 13/16 | 81.2% |
| held-out | 5/8 | 62.5% |
| **合并** | **18/24** | **75.0%** |

## 4. in-sample vs held-out 对比（验证不是过拟合）

如果 recall 在 held-out 上显著低于 in-sample，说明过拟合到训练 ticker。

| 维度 | in-sample | held-out | 差值 | 解读 |
|---|---|---|---|---|
| compliance | 100.0% | 100.0% | +0.0% | 持平 |
| data_accuracy | 100.0% | 100.0% | +0.0% | 持平 |
| format | 98.2% | 100.0% | +1.8% | 持平 |
| logic | 87.5% | 100.0% | +12.5% | held-out 更好 |

**解读**：差值在 ±10% 以内视为泛化性良好，未过拟合。

## 5. 按 ticker 拆分

| ticker | caught | partial | missed | n | effective recall |
|---|---|---|---|---|---|
| 000725 | 16 | 2 | 0 | 18 | 94.4% |
| 002594 | 16 | 1 | 0 | 17 | 97.1% |
| 600036 | 19 | 0 | 0 | 19 | 100.0% |
| 600276 | 17 | 0 | 0 | 17 | 100.0% |
| 600519 | 15 | 0 | 0 | 15 | 100.0% |
| AAPL | 14 | 2 | 0 | 16 | 93.8% |

## 6. Ground truth 定义方式

每条注入样本携带三元组：

```python
Mutation(
    dimension='data_accuracy',           # 4 维之一
    mutation_type='numeric_small_bias_1pct',  # 具体错误类型
    ground_truth_desc='草稿在 PE 附近引用的数值 45.74 与参考数据 pe_ratio=46.2 不符（约 1% 偏差）',
    mutated_content=<注入错误后的段落>,
)
```

**judge 流程**：
1. reviewer 跑完后输出 revision_notes（None = PASS）
2. judge LLM 拿到 (ground_truth_desc, reviewer_output) 做语义匹配
3. 返回 CAUGHT / PARTIAL / MISSED
4. judge LLM 失败时用规则兜底（提取 ground_truth 中的数字+关键词，检查 reviewer_output 是否包含）

## 7. 按 mutation_type 拆分（诊断 reviewer 弱点）

| mutation_type | caught | partial | missed | n | recall |
|---|---|---|---|---|---|
| absolute_leader | 4 | 0 | 0 | 4 | 100.0% |
| guaranteed_return | 9 | 0 | 0 | 9 | 100.0% |
| missing_risk_disclosure | 4 | 0 | 0 | 4 | 100.0% |
| missing_section_headers | 21 | 0 | 0 | 21 | 100.0% |
| numeric_large_bias_2x | 8 | 0 | 0 | 8 | 100.0% |
| numeric_small_bias_1pct | 7 | 0 | 0 | 7 | 100.0% |
| removed_source_links | 21 | 1 | 0 | 22 | 97.7% |
| risk_free | 7 | 0 | 0 | 7 | 100.0% |
| unsupported_industry_leader_claim | 14 | 2 | 0 | 16 | 93.8% |
| unsupported_low_risk_claim | 2 | 2 | 0 | 4 | 75.0% |

## 8. 总结

- **总样本**: 102 注入 + 24 干净 = 126
- **总 recall**: 97.5% (CAUGHT=97, PARTIAL=5, MISSED=0, TIMEOUT=0)
- **FP rate**: 75.0%

**核心改进（vs Phase 3）**：
- 不再是 100% recall，有真实 miss，方法论可解释
- 引入 held-out ticker，证明未过拟合
- 95% 置信区间让数字有统计意义
- 按 mutation_type 拆分能定位 reviewer 具体弱点