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

## 快速开始（Docker Compose）

仓库自带一份本地样例（`sample/source/`，步数 7459、总长 10007、权重 3 段 / 动量 3 段不同切分）：

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

如需重新生成样例：`python3 sample/make_sample.py sample/source`

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
```

## 设计要点

- **均分**：rank `i` 得 `[i*N//W, (i+1)*N//W)`，连续、全覆盖、大小差 ≤ 1。
- **崩溃安全**：数据先写 `*.tmp` 再 `os.replace` 原子改名；`shard.json` 最后写，
  作为分片完成标记；`job.json` 同样原子替换。服务重启时把 `running` 任务标记为
  可续做的 `failed`。
- **复用判定**：分片目录里标记文件存在且两个数据文件大小精确匹配 → 视为完整，
  续做时直接跳过（不重写）。
- **源变更防护**：续做前重新校验清单，步数/总长/区间与任务不一致则失败。
- **下载闸门**：仅当任务状态为 `done`（全部 rank 齐备）才放行下载，否则 409。
- **幂等任务**：`job_id` 由（源目录、清单内容、新进程数）哈希得出，重复发起
  同一请求返回同一任务。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

覆盖：均分数学、清单校验（缺口/重叠/大小/路径穿越）、端到端字节级一致性、
中断续做（完整分片不重写）、重启后复用、源变更检测、下载闸门、API 全流程。

## 已知限制

- 单进程服务：任务以线程方式跑在一个 uvicorn worker 内，请勿开多 worker。
- 下载按文件提供（weights/momentum 分开），未提供打包归档接口。
- 源文件在任务创建后被**同尺寸同步数**篡改无法识别（未做内容校验和）。
- `new_world_size > total_length` 会被拒绝（每个进程至少分到一个元素）。
