# Checkpoint Reshard API

训练进程数变化时，把同一份 float32 检查点（**权重 + 动量**）从只读源卷重切到新的进程数，
输出每个进程的全局下标区间和原始步数，供训练作业直接续跑。

- 源清单（`manifest.json`）声明步数、总长度、每个 float32 文件覆盖的全局下标区间；
  权重与动量的切分点**可以不同**。
- 源区间有**缺口或重叠**（或文件大小不符）→ 拒绝转换（HTTP 422）。
- 按新进程数把全局下标 `[0, total_length)` 均分为连续区间（大小差 ≤ 1 个元素），
  分块流式读写（默认 4 MiB，可用 `CHUNK_BYTES` 调整）。
- 每个 rank 目录同时产出该区间的 `weights.f32` 和 `momentum.f32`，外加
  `shard.json`（区间 + 原始步数）。
- 中断/重启后续做：**已完整落盘的分片原样复用**，只补未完成的；
  **整组齐备前下载一律 409**。
- **训练步数对账**：从旧检查点重跑后，登记运行（run）并接收调度器上报
  （尝试号、起跑检查点、最高完成步、可选的新产出检查点），随时查询任一已登记
  检查点里**每次尝试各有多少步被保留 / 未被保留**。

## 快速开始（Docker Compose）

仓库自带本地样例（`sample/`，均为总长 10007、权重 3 段 / 动量 3 段不同切分）：

- `sample/source/` —— 步数 **7459** 的源检查点（默认挂载源）；
- `sample/ckpt-8000/`、`sample/ckpt-8200/` —— 模拟训练跑到 8000 / 8200 步后
  新产出并重切好的检查点，用于演示步数对账。

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

如需重新生成样例：

```bash
python3 sample/make_sample.py sample/source            # step 7459
python3 sample/make_sample.py sample/ckpt-8000 --step 8000
python3 sample/make_sample.py sample/ckpt-8200 --step 8200
```

不用 Docker 本地跑：

```bash
pip install -r requirements.txt
SOURCE_DIR=$PWD/sample/source OUTPUT_ROOT=$PWD/outputs \
  uvicorn app.main:create_app --factory --port 8000
```

## API（均可在 /docs 中操作）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/reshard` | 发起转换：`{"new_world_size": 6}`；相同请求幂等返回同一任务 |
| GET | `/jobs` / `/jobs/{job_id}` | 任务列表 / 进度（含各 rank 区间、步数、完成数） |
| POST | `/jobs/{job_id}/resume` | 失败/重启后续做；已完成分片自动跳过 |
| GET | `/jobs/{job_id}/shards/{rank}/meta` | 该 rank 的区间 `[start, end)` 与原始步数 |
| GET | `/jobs/{job_id}/shards/{rank}/download?part=weights\|momentum` | 下载分片；任务未全部完成时 409 |
| POST | `/runs` | 登记运行及起始检查点：`{"run_id", "start_checkpoint": {"checkpoint_id", "job_id"}}`；相同请求幂等 |
| GET | `/runs` / `/runs/{run_id}` | 运行列表 / 详情（检查点 + 各尝试进度） |
| POST | `/runs/{run_id}/reports` | 调度器上报：`{"attempt", "start_checkpoint", "highest_step", "produced": [...]}`；整份原子提交 |
| GET | `/runs/{run_id}/checkpoints/{checkpoint_id}/reconciliation` | 沿实际产出链对账：各次尝试保留 / 未保留的步数 |

```bash
# 发起（source_dir 缺省用挂载的 SOURCE_DIR，也可显式覆盖）
curl -X POST localhost:8000/reshard -H 'Content-Type: application/json' \
     -d '{"new_world_size": 6}'
# 查询 → 拿到 job_id 与每个 rank 的 [start, end) 和 step
curl localhost:8000/jobs/<job_id>
# 中断后续做
curl -X POST localhost:8000/jobs/<job_id>/resume
# 整组完成后下载
curl -OJ "localhost:8000/jobs/<job_id>/shards/0/download?part=weights"
```

## 训练步数对账

从旧检查点重跑训练后，值班员需要知道：**选定的检查点里，每次尝试各有多少步
被保留、多少步白跑了**。模型与规则：

- **检查点**引用一个**已完成**的重分片任务（`job_id`）；检查点的步数就是该任务
  清单里的步数（单一事实来源，客户端不单独传步数）。
- **运行（run）**登记一个起始检查点；同一 `run_id` 重复登记同一起始检查点幂等，
  起始检查点不同则 409。
- **上报**：调度器按尝试上报 `attempt`（尝试号）、`start_checkpoint`（起跑检查点）、
  `highest_step`（最高完成步），可附 `produced`（新产出的检查点，各自引用一个
  已完成的重分片任务）。
  - 尝试只能从**本次运行已登记**的检查点起跑（否则 422）；
  - **迟到上报不能拉低进度**：已记录的最高完成步只增不减（取 max）；
  - 产出步数必须 **> 起跑步** 且 **≤ 本次上报的最高完成步**（越界 422）；
  - 起跑点或产出归属与已有记录**冲突**（同一尝试换起跑点、同一检查点被不同
    尝试/不同任务认领）→ 409；
  - 任何一条违规 → **整份上报拒收**，不产生部分生效；内容完全相同的重发是
    幂等重放，原样接受。
- **对账**：任选已登记的检查点，沿实际产出链（检查点 → 产出它的尝试 → 该尝试的
  起跑检查点 → … → 运行起点）计算：
  - 链上尝试：到它在链上产出的检查点为止的步数**保留**，超出部分**未保留**；
  - 链外尝试：步数**全部未保留**；
  - 同一步被多次重跑，**按尝试分别计数**（不做跨尝试去重）。
- 上报与查询交错时，每次对账只使用**同一个已提交状态**（单锁串行化提交与读取，
  每次提交原子落盘 `run.json`）。

### /docs 演示（本地样例）

```bash
# 1) 三个检查点各做一次重切（7459 用默认源；8000/8200 显式覆盖 source_dir）
curl -X POST localhost:8000/reshard -H 'Content-Type: application/json' -d '{"new_world_size": 6}'
curl -X POST localhost:8000/reshard -H 'Content-Type: application/json' \
     -d '{"new_world_size": 6, "source_dir": "/data/sample/ckpt-8000"}'
curl -X POST localhost:8000/reshard -H 'Content-Type: application/json' \
     -d '{"new_world_size": 6, "source_dir": "/data/sample/ckpt-8200"}'
# 记下三个 job_id：JOB_7459 / JOB_8000 / JOB_8200

# 2) 登记运行，起始检查点 ckpt-7459
curl -X POST localhost:8000/runs -H 'Content-Type: application/json' \
     -d '{"run_id": "demo", "start_checkpoint": {"checkpoint_id": "ckpt-7459", "job_id": "JOB_7459"}}'

# 3) 尝试 1：7459 → 8000，产出 ckpt-8000
curl -X POST localhost:8000/runs/demo/reports -H 'Content-Type: application/json' \
     -d '{"attempt": 1, "start_checkpoint": "ckpt-7459", "highest_step": 8000,
          "produced": [{"checkpoint_id": "ckpt-8000", "job_id": "JOB_8000"}]}'
# 4) 尝试 2：8000 → 8200，产出 ckpt-8200
curl -X POST localhost:8000/runs/demo/reports -H 'Content-Type: application/json' \
     -d '{"attempt": 2, "start_checkpoint": "ckpt-8000", "highest_step": 8200,
          "produced": [{"checkpoint_id": "ckpt-8200", "job_id": "JOB_8200"}]}'
# 5) 分叉：尝试 3 从 ckpt-7459 重跑，到 7700 挂了（无产出）
curl -X POST localhost:8000/runs/demo/reports -H 'Content-Type: application/json' \
     -d '{"attempt": 3, "start_checkpoint": "ckpt-7459", "highest_step": 7700}'
# 6) 迟到上报：尝试 2 自称最高 8100 —— 已记录的 8200 不会被拉低
curl -X POST localhost:8000/runs/demo/reports -H 'Content-Type: application/json' \
     -d '{"attempt": 2, "start_checkpoint": "ckpt-8000", "highest_step": 8100}'

# 7) 对账：选定 ckpt-8200 —— 尝试 1 保留 541 步、尝试 2 保留 200 步、
#    尝试 3（链外）241 步全部未保留
curl localhost:8000/runs/demo/checkpoints/ckpt-8200/reconciliation
# 选定旧检查点 ckpt-7459 —— 所有尝试的步数全部未保留（重跑场景）
curl localhost:8000/runs/demo/checkpoints/ckpt-7459/reconciliation
```

第 6 步的迟到上报与已记录内容一致（起跑点相同、不越界），会被接受，但已记录的
最高完成步保持 8200 不变；若把起跑点改成别的检查点则会 409。

## 源清单格式

```json
{
  "step": 7459,
  "total_length": 10007,
  "weights":  [{"path": "weights/part-0.f32",  "start": 0,    "end": 3500}, ...],
  "momentum": [{"path": "momentum/part-0.f32", "start": 0,    "end": 2000}, ...]
}
```

- 区间为**左闭右开**的全局元素下标；每类的所有区间必须**无缝无重叠**地铺满
  `[0, total_length)`，否则 422。
- 每个文件必须是**原始 little-endian float32**，大小恰好 `(end-start)*4` 字节。
- `path` 相对源目录，拒绝绝对路径与 `..`。

## 输出布局

```
outputs/<job_id>/
  job.json                 # 任务状态，每完成一个分片落盘一次
  rank_00000/
    weights.f32            # 区间 [start, end) 的权重
    momentum.f32           # 同一区间的动量
    shard.json             # {"rank","start","end","step",...} —— 最后写入的完成标记
outputs/runs/<run_id>/
  run.json                 # 运行状态（检查点登记 + 各尝试进度），每次提交原子落盘
```

## 设计要点

- **均分**：rank `i` 得 `[i*N//W, (i+1)*N//W)`，连续、全覆盖、大小差 ≤ 1。
- **崩溃安全**：数据先写 `*.tmp` 再 `os.replace` 原子改名；`shard.json` 最后写，
  作为分片完成标记；`job.json` / `run.json` 同样原子替换。服务重启时把
  `running` 任务标记为可续做的 `failed`。
- **复用判定**：分片目录里标记文件存在且两个数据文件大小精确匹配 → 视为完整，
  续做时直接跳过（不重写）。
- **源变更防护**：续做前重新校验清单，步数/总长/区间与任务不一致则失败。
- **下载闸门**：仅当任务状态为 `done`（全部 rank 齐备）才放行下载，否则 409。
- **幂等任务**：`job_id` 由（源目录、清单内容、新进程数）哈希得出，重复发起
  同一请求返回同一任务。
- **对账一致性**：`RunStore` 单锁串行化「校验 + 提交」与「查询」，每份上报要么
  整体提交要么整体拒收；对账结果只来自某一个已提交状态，不会读到两份上报
  之间的中间态。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

覆盖：均分数学、清单校验（缺口/重叠/大小/路径穿越）、端到端字节级一致性、
中断续做（完整分片不重写）、重启后复用、源变更检测、下载闸门、API 全流程；
步数对账（登记/上报校验、整份拒收、迟到上报、幂等重放、分叉链对账、
同一步重跑分别计数、重启后状态保留、上报与查询并发交错）。

## 已知限制

- 单进程服务：任务以线程方式跑在一个 uvicorn worker 内，请勿开多 worker。
- 下载按文件提供（weights/momentum 分开），未提供打包归档接口。
- 源文件在任务创建后被**同尺寸同步数**篡改无法识别（未做内容校验和）。
- `new_world_size > total_length` 会被拒绝（每个进程至少分到一个元素）。
- 对账只记录步数归属，不校验检查点内容；已完成的重分片任务若被外部删除，
  已登记的运行/检查点记录仍保留（步数事实不变），但下载会 404/409。

