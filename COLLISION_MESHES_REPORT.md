# linkerhand_l20 碰撞网格生成报告

**任务**：用 Blender 无头模式为 `linkerhand_l20` 灵巧手生成真正的碰撞网格（凸包），替换原先直接拷贝视觉网格的 `meshes/collisions/*.STL`。

**结果**：22 个带碰撞体的 link 全部生成完毕，独立校验全部通过（`RESULT: ALL CHECKS PASSED`）。
总三角面 **321,173 → 67,474（4.76×）**，22 个 mesh 逐个重建耗时合计 19.7 s（整次 Blender 进程 ~24 s）。
原始文件已备份到 `backup/collisions_original/`。

---

## 1. 交付物与复现方式

| 文件 | 作用 |
|------|------|
| `tools/make_collision_meshes.py` | 参数化 Blender 处理脚本（本次主交付） |
| `tools/convex_hull_np.py` | 纯 numpy float64 QuickHull + 平面/凸性/膨胀工具（脚本内部依赖） |
| `tools/validate_collisions.py` | 独立校验脚本（trimesh，不复用生成代码） |
| `tools/hull_fidelity.py` | 凸包保真度复核脚本（凸包“填掉了多少凹陷”） |
| `tools/test_convex_hull.py` | 自研凸包 vs scipy/Qhull 交叉验证（22 个真实点云完全一致） |
| `tools/render_preview.py` | 预览渲染（visual / collision 对照） |
| `logs/collision_build.json` | 生成日志（每 link 面数、迭代、穿模、耗时） |
| `logs/validation.json` / `logs/validation_table.md` / `.txt` | 独立校验数据 / markdown 表格 / 文本表格 |
| `logs/hull_fidelity.json` | 保真度数据 |
| `logs/previews/montage_<link>.png` | 每 link 四视角 visual(上) / collision(下) 对照图 |

复现全部结果（无 GUI，全程 headless）：

```bash
cd /home/chenyc/codes/linkerhand_l20

# 1) 生成碰撞网格（默认写入 URDF 指向的 meshes/collisions/，并备份原件）
/home/chenyc/Blender/blender-5.1.2-linux-x64/blender --background --factory-startup \
    --python tools/make_collision_meshes.py -- \
    --urdf linkerhand_l20.urdf --grid-mm 1.0 --margin-um 1.0 \
    --log logs/collision_build.json --backup-dir backup/collisions_original

# 2) 独立校验（trimesh 环境，仅用于校验）
/home/chenyc/.conda/envs/graspqp/bin/python tools/validate_collisions.py \
    --urdf linkerhand_l20.urdf --out-json logs/validation.json --md

# 3) 保真度复核 + 预览图
/home/chenyc/.conda/envs/graspqp/bin/python tools/hull_fidelity.py \
    --urdf linkerhand_l20.urdf --out-json logs/hull_fidelity.json
/home/chenyc/Blender/blender-5.1.2-linux-x64/blender --background --factory-startup \
    --python tools/render_preview.py -- --urdf linkerhand_l20.urdf --out-dir logs/previews
```

试运行不写盘：在 1) 中加 `--dry-run`；只处理部分 link：`--only index_distal,hand_base_link`。

**环境**：Blender 5.1.2（自带 Python 3.13 + numpy 2.3.4，无 scipy）；校验环境 conda `graspqp`（numpy 1.26.4 / trimesh 4.8.0 / scipy 1.15.3）。
生成流程只依赖 Blender 自带 numpy，**未向 `graspqp` 安装任何包**。

---

## 2. 方法

对每个 link：

1. **导入**视觉 STL（`bpy.ops.wm.stl_import`），合并成一个 mesh，按 `remove_doubles 1e-7` 焊接 → 得到视觉点云（带原始顶点）。
2. **体素抽稀**（`--grid-mm 1.0`，1 mm 立方体，每格保留距点云质心最远的点，偏向外表面）→ 仅作为凸包计算的**加速种子**，不是最终形状。
3. **严格凸包 + 包含性增长**：对种子做自研 float64 QuickHull，然后检查所有视觉顶点是否在凸包内（解析平面判据）；把仍在外部**的真实视觉顶点**加入输入重新求凸包，直到全部视觉顶点都在内部。
   - 收敛即“视觉网格的精确凸包”：**凸包的所有顶点都是真实的视觉顶点**，没有任何人造膨胀面。
   - 收敛判据 1 µm（`--margin-um`），实际收敛到 0.000–0.051 µm。
4. **拓扑整理**（bmesh）：焊接 1e-9、`dissolve_degenerate`、`recalc_face_normals`，若体积为负则整体翻转 → 统一外向绕序。
5. **导出** STL 二进制（`stl_export`，`global_scale=1.0`、`use_scene_unit=False`、`apply_modifiers=True`），路径/文件名严格取自 URDF 的 `<collision><geometry><mesh filename>`。
6. **导出后验证**（同一进程内重新导入导出结果）：水密、连通分量、体积符号、逐三角面平面判据穿模量、bbox 超出量。

### 为什么不用 Blender 自带的 `bmesh.ops.convex_hull`

在 1e4–2e4 点的真实点云上，Blender 的 `convex_hull` 返回**自相矛盾的多面体**：手掌 3036 个面里有 454 个不是支撑面，局部非凸达 31.7 mm；扰动（0.1 µm–30 µm）无效，已交叉核对 F/V 索引确实正确。因此改为自研 QuickHull（`tools/convex_hull_np.py`，纯 numpy float64），并用 `tools/test_convex_hull.py` 与 scipy/Qhull 在全部 22 个真实点云上逐项比对：顶点数、面数、体积一致（相对误差 < 1e-4），`hull_is_convex` 最差 0.0000 µm。

另一个曾导致严重错误的问题：面法向若按三角形索引顺序求，可能是朝内的，导致“包含性”判据整体反向（手掌被误判为要外扩 39 mm）。`facets_planes()` 现在用内部点统一校正法向（`if n@c > d: flip`），已修复。

---

## 3. 逐 link 结果

列说明：`视觉面` = 视觉 STL 三角面数；`视觉点` = 焊接后顶点数；`凸包输入点` = 最终凸包使用的点数（全部为视觉顶点）；`碰撞面` = 生成的碰撞网格三角面数；`比` = 视觉面/碰撞面；`迭代` = 包含性增长轮数；`穿模` = 视觉顶点最深穿出碰撞面的距离（µm，0 表示完全包含）；`凸包体积/包围盒` = 形状饱满度；`最大间隙` = 视觉表面到碰撞表面的最大距离（即凸包填掉的凹陷深度）。
| 1 | `hand_base_link` | 42908 | 21001 | 13907 | 3854 | 11.1&times; | 1 | Y | Y | 1 | 0.001 | 692260 | 0.556 | 18.63 | 1.37 |
| 2 | `pinky_metacarpals` | 1414 | 703 | 489 | 444 | 3.2&times; | 1 | Y | Y | 1 | 0.001 | 2695 | 0.602 | 4.60 | 0.12 |
| 3 | `pinky_proximal` | 16136 | 8070 | 2832 | 1022 | 15.8&times; | 15 | Y | Y | 1 | 0.026 | 12992 | 0.666 | 8.33 | 1.51 |
| 4 | `pinky_middle` | 14938 | 7462 | 2585 | 2184 | 6.8&times; | 1 | Y | Y | 1 | 0.001 | 6987 | 0.602 | 3.20 | 0.43 |
| 5 | `pinky_distal` | 19016 | 9496 | 4474 | 6904 | 2.8&times; | 1 | Y | Y | 1 | 0.002 | 5725 | 0.507 | 7.38 | 1.41 |
| 6 | `ring_metacarpals` | 1414 | 703 | 489 | 444 | 3.2&times; | 1 | Y | Y | 1 | 0.001 | 2695 | 0.602 | 4.60 | 0.11 |
| 7 | `ring_proximal` | 16136 | 8070 | 2832 | 1024 | 15.8&times; | 15 | Y | Y | 1 | 0.026 | 12992 | 0.666 | 8.33 | 1.50 |
| 8 | `ring_middle` | 14940 | 7463 | 2584 | 2192 | 6.8&times; | 1 | Y | Y | 1 | 0.001 | 6987 | 0.602 | 3.20 | 0.41 |
| 9 | `ring_distal` | 19016 | 9496 | 4474 | 6904 | 2.8&times; | 1 | Y | Y | 1 | 0.002 | 5725 | 0.507 | 7.38 | 1.38 |
| 10 | `middle_metacarpals` | 1414 | 703 | 489 | 444 | 3.2&times; | 1 | Y | Y | 1 | 0.001 | 2695 | 0.602 | 4.60 | 0.11 |
| 11 | `middle_proximal` | 16142 | 8073 | 2832 | 1024 | 15.8&times; | 15 | Y | Y | 1 | 0.026 | 12992 | 0.666 | 8.33 | 1.53 |
| 12 | `middle_middle` | 14938 | 7462 | 2585 | 2190 | 6.8&times; | 1 | Y | Y | 1 | 0.001 | 6987 | 0.602 | 3.20 | 0.42 |
| 13 | `middle_distal` | 19016 | 9496 | 4474 | 6902 | 2.8&times; | 1 | Y | Y | 1 | 0.002 | 5725 | 0.507 | 7.42 | 1.38 |
| 14 | `index_metacarpals` | 1414 | 703 | 489 | 444 | 3.2&times; | 1 | Y | Y | 1 | 0.001 | 2695 | 0.602 | 4.60 | 0.12 |
| 15 | `index_proximal` | 16140 | 8072 | 2832 | 1028 | 15.7&times; | 15 | Y | Y | 1 | 0.026 | 12992 | 0.666 | 8.33 | 1.50 |
| 16 | `index_middle` | 14938 | 7462 | 2584 | 2198 | 6.8&times; | 1 | Y | Y | 1 | 0.001 | 6987 | 0.602 | 3.20 | 0.43 |
| 17 | `index_distal` | 19016 | 9496 | 4474 | 6904 | 2.8&times; | 1 | Y | Y | 1 | 0.002 | 5725 | 0.507 | 7.57 | 1.40 |
| 18 | `thumb_metacarpals_base2` | 9650 | 4809 | 2203 | 1048 | 9.2&times; | 1 | Y | Y | 1 | 0.001 | 33784 | 0.547 | 9.41 | 0.26 |
| 19 | `thumb_metacarpals_base1` | 5732 | 2860 | 1506 | 812 | 7.1&times; | 1 | Y | Y | 1 | 0.001 | 13373 | 0.470 | 8.65 | 0.20 |
| 20 | `thumb_metacarpals` | 20932 | 10383 | 5725 | 7330 | 2.9&times; | 1 | Y | Y | 1 | 0.001 | 54279 | 0.683 | 14.62 | 1.60 |
| 21 | `thumb_proximal` | 13530 | 6701 | 3627 | 4196 | 3.2&times; | 1 | Y | Y | 1 | 0.001 | 12893 | 0.642 | 8.54 | 0.80 |
| 22 | `thumb_distal` | 22393 | 11078 | 5512 | 7982 | 2.8&times; | 1 | Y | Y | 1 | 0.051 | 8566 | 0.586 | 8.83 | 1.67 |

合计：视觉 321,173 面 → 碰撞 67,474 面（4.76×）；最差穿模 0.051 µm（`thumb_distal`，float32 STL 在该尺度下的分辨率约 0.02 µm）；全部 22 个 mesh 水密、绕序一致外向、单连通分量、欧拉示性数 2（亏格 0）。

`iteration=15` 的 4 个 link（`pinky/ring/middle/index_proximal`）是同一几何体的四个副本（4 指近节完全同形）。它们需要 15 轮点插入才收敛，末轮残余 68 nm 已低于该尺度 STL 的 float32 精度，属数值残差而非形状缺陷。

---

## 4. URDF 修改情况

**URDF 未作任何修改**（`linkerhand_l20.urdf` 与处理前逐字节相同，已用 `diff` 校验）。

理由：URDF 中 22 个 `<collision><geometry><mesh filename="meshes/collisions/<name>.STL"/>` 原本就全部存在、路径正确、`scale` 为 `1 1 1`、`origin` 全零，生成脚本按这些路径原位覆盖写入，文件名/格式/相对路径与 URDF 完全一致，因此无需改动。

两点说明（不影响 URDF）：
- `meshes/pinky_middle_f.STL` 与 `meshes/collisions/pinky_middle_f.STL` 在 URDF 中**未被任何 link 引用**，按“不改视觉网格”的要求原样保留（其 collision 副本因此仍是视觉网格的拷贝）。若确认废弃，可删除；若要启用，需在 URDF 中新增 link。
- `linkerhand_l20_root`、`root_link` 两个 link 没有 visual/collision，脚本自动跳过。

回滚：`cp backup/collisions_original/*.STL meshes/collisions/`。

---

## 5. 独立校验结果

`tools/validate_collisions.py` 完全独立于生成脚本（重新解析 URDF、用 trimesh 重新读盘），逐 mesh 检查：水密（2-流形、无边界边）`is_watertight`、绕序一致 `is_winding_consistent`、连通分量 `body_count`、体积符号、视觉顶点到碰撞面的穿模、碰撞面到视觉面的最大间隙、bbox 对比。

```
links checked 22, failing 0
RESULT: ALL CHECKS PASSED
```

完整 markdown 表格：`logs/validation_table.md`（对齐文本版 `logs/validation_table.txt`），原始数据 `logs/validation.json`。

> 视觉面数在本报告表格（Blender 导入统计）与校验表格（trimesh 读盘统计）间有 0–57 面差异（321,173 vs 321,230），来自两个加载器对重复顶点/退化面的焊接策略不同；碰撞面数两侧完全一致（67,474）。

---

## 6. 需要人工确认的 link

> **决策记录（已确认）**：面数模式 = 全部使用**精确凸包**（不用 coarse 模式）；`hand_base_link` = **保持单一凸包**。
> 因此 §6.1 方案 1、§6.2 现状即为最终交付状态，`meshes/collisions/` 中文件无需再改；以下保留各候选方案的量化依据，供日后需要时参考。

按“是否需要拍板”分三类，每类给出数字依据与建议。

### 6.1 手掌 `hand_base_link`：凸包把掌心凹陷填成了一块实体（**建议关注**）

- 数据：碰撞面 3854（11.1×），凸包体积 692,260 mm³（体积/包围盒 = 0.556）；视觉表面只有 11.0% 落在凸包表面上，视觉表面到碰撞面的距离中位数 2.81 mm、95 分位 9.80 mm、**最大 18.63 mm**；bbox 超出 0。
- 原因：手掌在拇指根与四指根之间有一处深达 ~19 mm 的掌窝/斜面，凸包必然把它封平。这不是实现缺陷，是“凸包”这一形状定义的必然结果。
- 影响：碰撞检测中掌心区域变成实心块，抓握时物体无法“贴进”掌窝；`hand_base_link` 与手指之间不会有碰撞（自碰撞白名单），所以通常无害，但**若你的抓取管线依赖掌心凹陷**，单凸包就是错的表现形式。
- 候选方案（需你选）：
  1. 保持现状（推荐，符合“形状 = 凸包”的原始要求）；
  2. 把 `hand_base_link` 拆成 2–3 个凸块（例如掌体 + 拇指根楔形块），仍全部是凸包，但需要新增 link/joint 或改用 `<collision>` 多 mesh 拼接——**会改动 URDF**，超出本次“不改 URDF”的约束；
  3. 改用 `convex decomposition`（V-HACD），得到非凸碰撞体，面数会显著上升且不再是凸包。

### 6.2 面数超出“几千面”建议值的 6 个 link

| link | 碰撞面 | 视觉面 | 倍数 |
|------|-------:|-------:|-----:|
| `thumb_distal` | 7982 | 22414 | 2.8× |
| `thumb_metacarpals` | 7330 | 20932 | 2.9× |
| `pinky_distal` / `ring_distal` / `index_distal` / `middle_distal` | 6902–6904 | 19016 | 2.8× |

- 原因：这些件是**曲面 + 密集褶皱**的壳体（视觉网格为 1.9–2.2 万面的高精度曲面），凸包需要在弯曲方向上铺足够多面片才能贴合；每个凸包面片至少需要一个真实视觉顶点，所以面数下界由“褶皱峰数量”决定。其余 16 个 link 在 444–4196 面（3.2×–15.8×）：14 个 < 3000 面，2 个在 3854–4196 面（`hand_base_link`、`thumb_proximal`）。
- 我已实现并实测了“少面数”候选方案（`--mode coarse`：种子凸包 + 把面片推到真实支撑面），结果：总面数 24,032（13.4×），单 link 612–3196 面，但代价是
  - 相对精确凸包外扩 0.02–0.26 mm（bbox 超出 +0.25 mm），
  - 局部非凸 24.9–425 µm（顶点由近共面三平面求交，数值上过约束），
  - 22 个 link 中有 6 个仍残留 70–203 µm 的穿模（视觉顶点漏出凸包）。
  由于“必须完全包含、不能明显外扩”是硬约束、面数是建议值，**默认仍输出精确凸包**；若你更看重面数，可切 `--mode coarse`，但需要接受上述误差（我可以再迭代把 coarse 模式的穿模压到 0，代价是面数向精确凸包靠拢）。
- 另一个折中：只对这 6 个 link 用 coarse 模式，其余用精确凸包。**这项需要你拍板**。

### 6.3 已量化但判定为“无需处理”的现象

- **零面积面片**：`pinky/ring/middle/index_proximal`、`thumb_metacarpals` 各有 1 个面片高度 < 10 nm（另有 2–3 个面片面积 < 0.1 µm²），合计 5 个 link、7 个面片。成因是视觉网格在 CAD 直边上用了近共线顶点采样，凸包据此生成微米级面片。它们对水密性/绕序/体积/包含性均无影响（校验全通过）。曾尝试删除对应顶点，实测会让凸包内凹 214 µm（说明该顶点确实是支撑点），因此保留并在日志 `degenerate_hull_facets` 中记录。
- **凸包填掉的凹陷**：除手掌外，各 link 最大间隙 3.2–14.6 mm（bbox 对角线的 15%–19%），基本等于壳体内部空腔/螺柱的深度。这意味着碰撞壳是“外壳+空腔填实”的实心凸体，符合任务要求（填掉管腔、支撑、螺纹孔），但**不可用于需要空腔的内部装配干涉检查**。

---

## 7. 其它已知限制与实现细节

- **加密阈值**：`--grid-mm 1.0` 只影响种子速度，不影响最终形状（最终凸包由真实视觉顶点决定）。用更细的网格（0.5 mm）实测得到的是同一个凸包，只是迭代次数更少；用 2.0 mm 会略微增加迭代次数。默认 1.0 mm 是在 22 个 link 上总耗时最短的设置（~20 s）。
- **共面性容差**：凸性检查容差 1e-9（相对尺度），实际最差 0.0000 µm。
- **float32 分辨率**：STL 为 float32，0.17 m 尺度上最小可分辨位移约 20 nm。`*_proximal` 类的 68 nm 残差、`thumb_distal` 的 0.051 µm 穿模都属于该量级，不构成真实穿透。
- **不修改视觉网格**：脚本只读取 `meshes/*.STL`，写入目标仅限 `meshes/collisions/*.STL`（+ 备份目录）。
- **脚本参数**：`--urdf --out-dir --grid-mm --margin-um --log --only --backup-dir --ascii --dry-run --mode {exact,coarse} --offset-um`。默认写入 URDF 指定路径；`--out-dir` 可用于先写临时目录做对比。
- **日志**：脚本对每个 link 输出一行 `[ok] <link> tris … iters … pen … wt … comps … bbox+ … (Ns)`，异常 link 输出 `[problem] <link>: 原因`；完整 JSON 见 `logs/collision_build.json`，运行日志 `logs/build.log`。

## 8. 变更文件清单

| 路径 | 状态 |
|------|------|
| `meshes/collisions/*.STL`（22 个） | **已覆盖**为凸包碰撞网格 |
| `backup/collisions_original/*.STL`（22 个） | 原始文件备份（`pinky_middle_f.STL` 未被覆盖，故无需备份，原件仍在原位） |
| `linkerhand_l20.urdf` | **未改动** |
| `meshes/*.STL`（视觉网格） | **未改动** |
| `tools/*.py`（7 个，见 §1） | 新增 |
| `logs/*` | 生成日志、校验数据、预览图 |
