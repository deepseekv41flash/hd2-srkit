# srkit — Stingray 引擎资源解包/打包工具（Helldivers 2）

> 纯 Python 标准库实现，无第三方依赖，不需要 Blender。
> 在 **build 25988413** 上实测通过（17/17 自检）。

`srkit` 读写 Stingray 引擎的资源归档：`bundles.nxa` / legacy 三件套 / DSAR 单文件、`.patch_N`
补丁归档、以及 `unit`(mesh) / `texture` / `material` / `bones` / `lua` 等资源类型。

## 这个仓库 **不** 包含什么

明确划清边界，避免误导：

| 不包含 | 原因 |
|---|---|
| **运行期内存改写**（原 `memmod` 模块） | 新版游戏里 Lua 已无 `ffi`、且外部进程读内存被 EAC 拒绝 ⇒ 那条路已不可用，代码不再发布 |
| **Lua addon 构建管线**（原 `modkit` / `srkit mod`） | 它内含运行期内存模板与各 build 的 PE 签名，属于上面那条已失效的路 |
| 游戏资产 | 仓库只有代码与自检夹具，不含任何游戏文件 |
| 特定 build 的偏移补丁集 | 见下方「跨版本」 |

本仓库只做**资源层**：解包、解析、改字节、重新打包、往返校验。

## 安装

```bash
# 无依赖，直接跑
git clone <this repo> srkit && cd srkit
python -m srkit --help

# 或者装成命令（可选）
pip install -e .
srkit --help
```

需要 Python ≥ 3.10（仅标准库）。

## 快速开始

```bash
# 1) 看归档里有什么（自动识别 bundles.nxa / legacy 三件套 / DSAR）
python -m srkit info "D:/SteamLibrary/steamapps/common/Helldivers 2/data"

# 2) 列资源（可按类型过滤）
python -m srkit list "D:/HD2/data" --type mesh --type texture --limit 20

# 3) 解包成工作区（JSON 描述 + 原始 payload）
python -m srkit unpack "D:/HD2/data" -o ./ws --type texture --decode-vertices

# 4) 往返校验：解码 → 编码 → 与原始字节逐字节比较
python -m srkit verify ./ws

# 5) 改完重新打包（默认 legacy 三件套；--compress 输出 DSAR）
python -m srkit pack ./ws -o ./repacked

# 6) 重建成零售版 bundle 集合
python -m srkit rebuild-bundles ./repacked -o ./bundles

# 7) 单个资源
python -m srkit extract "D:/HD2/data" --id <resource-id> -o out.unit

# 8) 空 unit 模板 → 造新 unit（补位/换壳）
python -m srkit emptyunit --package <patch> --name mods/mymod/unit -o out.patch_0

# 9) 自检（不需要游戏文件）
python tests/selftest.py
```

## 能力一览

| 模块 | 能力 |
|---|---|
| `container/repo.py` | 识别并遍历 `data/` 下的归档形态；`.patch_N` 覆写语义 |
| `container/bundle.py` | `bundles.nxa` 索引与重建 |
| `container/dsar.py` | DSAR（LZ4 分块）读写 |
| `container/package.py` | toc 包解析/序列化；流大小修复；padding/对齐 |
| `container/stream_metadata.py` | `.stream` 段大小的解析与推断（Unit/Bones 归零、Texture 取 toc 体） |
| `container/unit_refs.py` | unit / material 内部引用的列举与重写 |
| `container/empty_unit.py` | 空 unit 模板抽取、克隆、审计、消毒 |
| `resources/*` | `unit`(mesh/顶点/骨骼)、`texture`、`material`、`bones`、`lua` 编解码，其余类型 raw 回退 |
| `util/ids.py` | Stingray 名称哈希（Murmur32/64）、类型名表、资源 ID 解析 |
| `util/hashing.py` | 引擎侧 djb2-33 等哈希 |
| `util/lz4block.py` | 纯 Python LZ4 block 压缩/解压 |
| `workspace.py` | 工作区布局、打包/解包/校验流水线 |

运行 `python -m srkit codecs` 可看当前注册的全部资源类型与是否可解码。

## 自检

```
$ python tests/selftest.py
ok    lz4 / dsar / package / bundles / texture / material / bones / ids / unit
ok    patch-layout / lua / real-variants / workspace / bundled-workspace
ok    stream-metadata / unit-refs / empty-unit
all checks passed      # 17 checks
```

全部离线、自带夹具，**不需要游戏文件**。

## 跨版本

资源格式是引擎层的，跨 build 基本稳定；但**内容**会变（类型表、流大小、补丁索引）。
本仓库的验证基线是 **build 25988413**。游戏更新后请先跑一遍 `python tests/selftest.py`，
再用 `info` / `verify` 对真实的 `data/` 做抽样。

## 相关

- 逆向笔记与数据表工具（同一 build）：**[hd2-25988413](https://github.com/deepseekv41flash/hd2-25988413)**
  —— 密封盒、内容哈希闸门、实体组件表、可选弹种等。

## 许可

MIT
