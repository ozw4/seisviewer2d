# seisviewer データ全体推論

## 1. 目的

選択した設定（pipeline/keys/weights/pick params）で、**全セクション（key1 全件）に対して**

- ノイズ抑制（denoise）
- 初動確率マップ（fbpick prob）
- 初動ピック（1D）

    を **バックグラウンドジョブで一括適用**し、成果物を **job_id 配下**へ保存・別画面で閲覧/エクスポートできるようにする。


---

## 2. UI/操作（別画面 or 別window）

別画面（例：`/batch`）で以下を選択可能にする：

### 選択項目

- 適用処理
    - pipeline 処理（既存の pipeline を使用。**bandpass を含む**）
    - denoise
    - fbpick（初動読み）
- primary key / secondary key（byte offset）
- 適用重み（UIは **ドロップダウン**、**表示としてパス**）
- pick 設定
    - Pick method: `expectation` / `argmax`
    - `sigma_ms_max`（閾値未満は無効ピック）
    - Snap（波形特徴に吸着）
        - 参照：**raw**
        - **既に実装済み関数を使用**
        - 特徴種類（モード）を選択可能
        - 窓幅を指定可能
    - Subsample（Snapと併用するサブサンプル補正、選択可能）
- denoise parameter（選択可能）

### 適用範囲

- **適用は全件**（key1 の全 unique 値）

---

## 3. 実行方式（バックグラウンドジョブ）

- 適用は **バックグラウンド処理可能**（job_id を返す）
- 進捗率：`processed_key1 / total_key1`
- ジョブの挙動
    - **1セクション失敗で全体停止**
    - GPU推論は **同時実行しない**
- 運用条件
    - サーバは **`-workers 1` に固定**

---

## 4. Pipeline 方針

- **既存の pipeline 処理を使用**（bandpass を含む）
- pipeline に **初動読み（fbpick）も加えたい**（pipeline step として扱う）
- キャッシュ無効化
    - **model_version（mtime 等）を含める**（重み更新で別扱いになるようにする）

---

## 5. 保存先と成果物

### 保存先

- **ジョブ成果物（job_id 配下）のみ**に保存（エクスポート前提）

### 保存するもの

1. **ノイズ抑制出力（3D npy）**
- dtype：**float32**
- ダウンサンプル：**しない**
- padding：**0埋め**
1. **初動確率マップ（3D npy）**
- dtype：**float16**
- ダウンサンプル：**しない**
- padding：**0埋め**
- shape（3D）: **(primary, secondary, time)**
1. **初動ピック（1D）**
- 保存：**1D pick を保存**
- 無効ピック：
    - `sigma_ms_max` に満たない（指定条件）ものは **無効ピック**
- **manual pick 出力 npz の期待フォーマットに合わせる**
    - 予測ピック npz のキー/構造を manual export と整合させる

---

## 6. 3D npy の軸定義・パディング規約（確定）

### 軸

- `primary`：**key1 の値の列（セクション列）**
    - 順序：**TraceStore の `np.unique(key1s)`**
- `secondary`：各セクション内の列（トレース方向）
    - パディングあり
- `time`：サンプル方向

### padding

- 3D本体（denoise/prob）: **0埋め**
- `key2_values_padded` の padding：**0**
- Pick が NaN のとき：**Snap は実行しない**

---

## 7. Snap の仕様（確定部分）

- Snap は **波形特徴に吸着**
- 参照：**raw**
- 実装：**既存の実装済み関数を使用**
- UIから指定できる：
    - 波形特徴種類
    - 窓幅

## 1) 成果物（job_id 配下に保存するファイル仕様）

保存先は既存と同じ `pipeline_jobs` 配下（`app/services/pipeline_artifacts.py:get_job_dir(job_id)`）を使い、**job_id ディレクトリ直下**にまとめます（既存 pipeline/all の `key1/<tap>.bin` とは衝突しない命名にする）。

`<PIPELINE_JOBS_DIR>/<job_id>/`

1. `job_meta.json`（JSON）
    - 入力：BatchApplyRequest をほぼそのまま＋解決済み model_version（mtime込み）
    - 例フィールド：
        - `file_id`, `key1_byte`, `key2_byte`
        - `pipeline_spec`（steps）
        - `outputs`（save_denoise/save_prob/save_picks）
        - `pick_options`（method/subsample/sigma_ms_max/snap設定）
        - `resolved_models`: `{ "denoise": {"id": "...", "version": "file:mtime_ns"}, "fbpick": {...} }`
        - `dt`, `n_samples`, `n_key1`, `max_traces_padded`
        - `padding`: `{ "trace_pad_value": 0, "key2_pad_value": 0 }`
        - `created_at`, `finished_at`
2. `key1_values.npy`
    - dtype: `int32`
    - shape: `(n_key1,)`
    - 並び：`TraceStoreSectionReader.get_key1_values()`（= `np.unique(key1s)`）
3. `key2_values_padded.npy`
    - dtype: `int32`
    - shape: `(n_key1, max_traces)`
    - 各 key1 の display order の key2 値を左詰め、残りは `0` padding
4. `denoise_f32_padded.npy`（outputs.save_denoise が true の時のみ）
    - dtype: `float32`（固定）
    - shape: `(n_key1, max_traces, n_samples)`
    - 各 key1 の denoise 出力（pipeline 内の denoise step 通過直後）を左詰め、残りは `0` padding
5. `fbpick_prob_f16_padded.npy`（outputs.save_prob が true の時のみ）
    - dtype: `float16`（固定）
    - shape: `(n_key1, max_traces, n_samples)`
    - fbpick probability map（pipeline 内の fbpick analyzer 出力の `prob`）を左詰め、残りは `0` padding
    - **downsample 機能なし**
6. `predicted_picks_time_s.npz`（outputs.save_picks が true の時のみ）
    - 既存 `app/api/routers/picks.py:export_manual_picks_npz()` とキー互換に寄せる
    - 収録（最低限）：
        - `picks_time_s`: `float32`, shape `(n_traces_total,)`（**original order** / invalid は `NaN`）
        - `n_traces`: `int64` or `int32`（np scalar）
        - `n_samples`: `int64` or `int32`
        - `dt`: `float32`
        - `format_version`: `"manual_picks_npz_v1"`（同名）
        - `exported_at`: ISO string
        - `export_app`: `"seisviewer2d"`
        - `source_hint`: `"batch_predicted_picks"`
    - 注意：**sigma_ms_max を満たさない（sigma > sigma_ms_max）ものは NaN**
    - 注意：**pick が NaN のとき snap は実行しない**

---

## 2) API 設計（新規エンドポイント）

### 2.1 追加：Batch Apply 用 router（新規）

**ファイル**: `app/api/routers/batch_apply.py`（新規）

**登録**: `app/api/routers/__init__.py` と `app/main.py` に include

### (1) `POST /batch/apply`

- 入力（JSON）: `BatchApplyRequest`（後述 schemas 参照）
- 出力（JSON）: `{"job_id": "<id>", "status": "queued"}`

### (2) `GET /batch/job/{job_id}/status`

- 出力（JSON）: `BatchJobStatusResponse`
    - `status`: `"queued"|"running"|"done"|"error"`
    - `progress`: `processed_key1 / total_key1`（0〜1）
    - `message`: 任意（例: `"processing key1=123 (5/80)"`）

### (3) `GET /batch/job/{job_id}/files`

- 出力（JSON）: 生成済みファイル一覧
    - `files: [{ "name": "fbpick_prob_f16_padded.npy", "size_bytes": 12345 }, ...]`

### (4) `GET /batch/job/{job_id}/download?name=<filename>`

- `FileResponse` で返す
- **path traversal 対策**：name は `Path(name).name == name` を必須、かつ job_dir 直下のみ許可

---

## 3) Request/Schema（Pydantic）追加

**ファイル**: `app/api/schemas.py`（追記）

### 3.1 `SnapOptions`

- `enabled: bool`
- `feature: Literal["peak","trough","upzc","maxgrad"]`
    - viewer_legacy.js の `pickSnapMode` 相当（`adjustPickToFeature()` の分岐に合わせる）
- `refine: Literal["none","parabolic","zc"]`
- `window_ms: float`（例 20.0）

### 3.2 `PickOptions`

- `method: Literal["expectation","argmax"]`
- `subsample: bool`（argmax のとき prob 上で parabolic 補間するか、expectation は常に float index）
- `sigma_ms_max: float | None`（None なら gate 無し）
- `snap: SnapOptions`

### 3.3 `BatchOutputs`

- `save_denoise: bool`
- `save_prob: bool`
- `save_picks: bool`

### 3.4 `BatchApplyRequest`

- `file_id: str`
- `key1_byte: int = 189`
- `key2_byte: int = 193`
- `pipeline_spec: PipelineSpec`
- `outputs: BatchOutputs`
- `pick_options: PickOptions`

### 3.5 `BatchApplyResponse`, `BatchJobStatusResponse`

- `BatchApplyResponse`: `job_id: str`, `status: str`
- `BatchJobStatusResponse`: `status: str`, `progress: float`, `message: str`

---

## 4) バックグラウンド実行（job runner）設計

### 4.1 新規 service

**ファイル**: `app/services/batch_apply_service.py`（新規）

### (A) `start_batch_apply_job(...) -> str`

- I/O:
    - 入力：`req: BatchApplyRequest`, `state: AppState`
    - 出力：`job_id: str`
- 処理：
    1. **spec 正規化**（model_version を params に注入、後述 5章）
    2. `job_id = uuid4().hex[:12]`
    3. `state.jobs[job_id] = {...}` を作成
        - `status="queued"`, `progress=0`, `message=""`
        - `file_id/key1_byte/key2_byte/pipeline_key/...`（必要なら）
        - `created_ts`
        - `artifacts_dir=str(get_job_dir(job_id))`
    4. `threading.Thread(target=_run_batch_apply_job, args=(job_id, req2, state), daemon=True).start()`

### (B) `_run_batch_apply_job(job_id: str, req: BatchApplyRequest, state: AppState) -> None`

- 例外ポリシー：**1セクション失敗で全体停止**（catch して job を `"error"` にして終了、続行しない）
- 処理フロー（重要ポイント込み）：
1. 初期化
- `reader = get_reader(file_id, key1_byte, key2_byte, state=state)`
- `key1_values = reader.get_key1_values()`（この順序が確定仕様）
- `dt = get_dt_for_file(file_id)`（`app/utils/segy_meta.py` 経由）
- `n_samples = reader.get_n_samples()`
1. `max_traces` 決定（padding 前提のため先に確定）
- 全 key1 を走査して `n_tr = len(reader.get_trace_seq_for_value(key1, align_to='display'))`
- `max_traces = max(n_tr)`
1. 出力 memmap を job_dir に作成（巨大化前提）
- `job_dir = get_job_dir(job_id)`
- `open_memmap(job_dir/"fbpick_prob_f16_padded.npy", dtype=np.float16, shape=(n_key1,max_traces,n_samples))` 等
- `key2_values_padded.npy` も `open_memmap(..., dtype=np.int32, shape=(n_key1,max_traces))`
- `picks_sorted = np.full((n_traces_total,), np.nan, dtype=np.float32)`（または memmap）
1. key1 ループ（進捗更新）
- `processed_key1 = 0`
- 各 `key1` で：
    - `raw = coerce_section_f32(reader.get_section(key1).arr)`（`app/services/reader.py` の `coerce_section_f32` を使う）
    - `key2_vals = reader.get_key2_values_for_section(key1, key2_byte)`（display order）
    - `trace_seq_display = reader.get_trace_seq_for_value(key1, align_to='display')`
    - `y = raw`
    - pipeline step 実行（既存 op をそのまま使用するが、中間を捕捉）
        - `for step in pipeline_spec.steps:`
            - `if step.kind == "transform":`
                - `y = TRANSFORMS[step.name](y, params=step.params, meta=meta)`
                - `if step.name == "denoise" and outputs.save_denoise: denoise_out = y`
            - `if step.kind == "analyzer" and step.name == "fbpick":`
                - `res = ANALYZERS["fbpick"](y, params=..., meta=meta)`
                - `prob = res["prob"]`（float32想定）→ `prob_f16 = prob.astype(np.float16)`
    - padding して memmap に格納
        - `key2_values_padded[i,:n_tr] = key2_vals; それ以外 0`
        - `denoise_f32_padded[i,:n_tr,:] = denoise_out; それ以外 0`
        - `prob_f16_padded[i,:n_tr,:] = prob_f16; それ以外 0`
    - pick 生成（outputs.save_picks && prob がある場合）
        - `picks_time = build_picks_from_prob(prob, dt, pick_options, raw_for_snap=raw)`
            - invalid は `NaN`
            - NaN のとき snap しない
        - `picks_sorted[trace_seq_display] = picks_time.astype(np.float32)`
    - `processed_key1 += 1`
    - `progress = processed_key1 / total_key1`
    - `state.jobs[job_id]["status"]="running"; ["progress"]=progress; ["message"]=...`
1. picks を original order に変換して npz 保存
- `sorted_to_original = reader.sorted_to_original`（TraceStoreSectionReader の既存属性）
- `picks_orig = np.full((n_traces_total,), np.nan, dtype=np.float32)`
- `picks_orig[sorted_to_original] = picks_sorted`
- `np.savez(job_dir/"predicted_picks_time_s.npz", picks_time_s=picks_orig, dt=..., n_samples=..., n_traces=..., format_version="manual_picks_npz_v1", ...)`
1. `job_meta.json` と `key1_values.npy` を保存
- `np.save(job_dir/"key1_values.npy", key1_values.astype(np.int32))`
- `job_meta.json` を write
1. 終了
- `state.jobs[job_id]["status"]="done"; finished_ts=...`

---

## 5) キャッシュ無効化（model_version を pipeline_key に含める）

要件 D「cache 無効化：model_version（mtime等）を含める」を満たすため、**pipeline_key の入力 spec の params に model_version を注入**します（`app/utils/pipeline.py:pipeline_key()` は params hash なので効く）。

### 5.1 新規：pipeline spec 正規化

**ファイル**: `app/services/pipeline_spec_normalize.py`（新規）

### `normalize_pipeline_spec_for_cache(spec: PipelineSpec) -> PipelineSpec`

- 入力：`PipelineSpec`
- 出力：`PipelineSpec`（deep copy して params を更新）
- ルール：
    - `denoise` step があるなら：
        - `model_id` を解決（未指定ならデフォルト）
        - `model_ver = model_version(path)`（`filename:mtime_ns`）
        - `step.params["_model_version"] = model_ver`
    - `fbpick` step があるなら：
        - 同様に `step.params["_model_version"]` を注入
    - 注入キーは `_model_version` のように先頭 `_` にして UI/既存 params と衝突しにくくする

        （ops 側は未知キーを無視する設計のままでOK）


### 5.2 呼び出し箇所（既存 router 修正）

**ファイル**: `app/api/routers/pipeline.py`（変更）

- `/pipeline/section` と `/pipeline/all` の入り口で
    - `spec2 = normalize_pipeline_spec_for_cache(spec)`
    - `pipeline_key = pipeline_key(spec2)`
    - `apply_pipeline(... spec=spec2 ...)`
- これで **pipeline tap cache** も **モデル更新で自動的に別キー**になります

---

## 6) denoise 重みの選択（dropdown用の model list と解決）

### 6.1 新規：denoise_models ヘルパ

**ファイル**: `app/utils/denoise_models.py`（新規）

- `denoise_model_dir() -> Path`（`Path(__file__).resolve().parents[2] / "model"`）
- `list_denoise_models() -> list[dict[str, str|bool]]`
    - glob 例：`"recon_*.pth"`（README の `recon_replace_edgenext_small.pth` に合わせる）
    - 返す dict 例：`{"id": "recon_replace_edgenext_small.pth", "path": ".../model/recon_replace_edgenext_small.pth"}`
- `validate_model_id(model_id: str) -> str`
- `resolve_model_path(model_id: str|None, default_id: str, require_exists: bool) -> Path`
- `model_version(path: Path) -> str`（fbpick と同仕様）

### 6.2 API：denoise_models 追加

**ファイル候補**

- (案1) `app/api/routers/pipeline.py` に `@router.get("/denoise_models")` を追加
- (案2) 新規 `app/api/routers/denoise.py` を作って include

**I/O**

- `GET /denoise_models` → `{"default_model_id": "...", "models":[{"id":"...","path":"..."},...]}`

---

## 7) GPU 推論の同時実行禁止（プロセス内ロック）

要件 E「GPU 推論は同時実行しない」を **router/job が並列に走っても守る**ため、推論箇所を lock で囲います（workers=1 前提なのでプロセス内 lock で十分）。

### 7.1 新規 lock

**ファイル**: `app/utils/gpu_lock.py`（新規）

- `GPU_INFER_LOCK = threading.Lock()`

### 7.2 適用箇所

**ファイル**: `app/utils/ops.py`（変更）

- `op_denoise(...)` 内で `with GPU_INFER_LOCK: denoise_tensor(...)`
- `op_fbpick(...)` 内で `with GPU_INFER_LOCK: infer_prob_map(...)`
    - これで `pipeline/all`、`fbpick_section_bin`、`batch/apply` が同時に走っても GPU は直列化されます

---

## 8) denoise を model_id で切替（backend）

### 8.1 denoise 本体改修

**ファイル**: `app/utils/denoise.py`（変更）

- 現状：固定パス `_DEFAULT_MODEL_PATH` + `_MODEL` singleton
- 変更方針：
    - `get_denoise_model(*, model_path: Path | None) -> tuple[torch.nn.Module, torch.device]`
        - `model_path is None` のときだけ既存 `_DEFAULT_MODEL_PATH`
        - **cache は model_path(or model_version) 単位**にする（dict）
    - `denoise_tensor(..., model_path: Path | None = None) -> np.ndarray`
        - dtype は常に float32 を返す（既存挙動踏襲）

### 8.2 pipeline op 側で model_id を読む

**ファイル**: `app/utils/ops.py`（変更）

- `op_denoise(x, params, meta)` に
    - `model_id = params.get("model_id")`
    - `model_path = resolve_model_path(model_id, require_exists=True)`（`denoise_models.py`）
    - `denoise_tensor(..., model_path=model_path)` を呼ぶ

---

## 9) Pick 生成（expectation/argmax + subsample + sigma gate + snap）

### 9.1 “予測 pick の生成”ロジックを router から分離

いま `app/api/routers/fbpick_predict.py` に `_chunked_expectations` / `_compute_picks` があるので、batch 側も使えるように service 化します。

**ファイル**: `app/services/fbpick_predict_service.py`（新規）

- `compute_pick_indices_from_prob(prob: np.ndarray, *, method: str, subsample: bool) -> np.ndarray`
    - 入力：
        - `prob`: `(n_traces, n_samples)` float32/float16（内部 float32で計算）
        - `method`: `"expectation"|"argmax"`
        - `subsample`: bool
    - 出力：`idx_float`: `(n_traces,)` float64（サンプル index）
- `compute_sigma_ms(prob: np.ndarray, dt: float) -> np.ndarray`
    - expectation 計算時の sigma を ms にして返す（既存ロジック踏襲）
- `apply_sigma_gate(idx, sigma_ms, sigma_ms_max) -> idx_gated`（超えたら NaN）

### 9.2 snap（raw 参照・既存JSロジックを Python に移植）

**ファイル**: `app/utils/snap.py`（新規）

- `snap_pick_to_feature(trace: np.ndarray, *, idx: float, mode: str, refine: str, window_samples: int) -> float`
    - 入力：
        - `trace`: `(n_samples,)` raw trace（float32）
        - `idx`: pick sample index（float）
        - `mode`: `"peak"|"trough"|"upzc"|"maxgrad"`
        - `refine`: `"none"|"parabolic"|"zc"`
        - `window_samples`: `round(window_ms/1000/dt)`（左右）
    - 出力：snap 後の sample index（float）
- `batch_apply_service` 側で：
    - pick が `NaN` のときは snap 関数を呼ばない（決め打ち仕様）

### 9.3 batch job 内の最終 pick（time秒）

**batch_apply_service 内部関数（同ファイル内 private でもOK）**

- `build_picks_time_s(prob, dt, pick_options, raw_for_snap) -> np.ndarray`
    - 出力：`(n_traces,) float32`（time秒、invalid NaN）

---

## 10) フロントエンド（別画面 / 別window）

### 10.1 新規ページ

**ファイル**: `app/static/batch_apply.html`（新規）

**ルーティング**: `app/main.py` に `@app.get("/batch", response_class=HTMLResponse)` を追加して返す

画面要素（最低限）：

- 適用処理：
    - checkbox: `save_denoise`, `save_prob`, `save_picks`
- primary/secondary key：
    - `key1_byte`, `key2_byte`（初期値は viewer の `window.currentKey1Byte/currentKey2Byte` を query に渡してセット）
- 重みパス選択（dropdown + パス表示）：
    - denoise: `/denoise_models` から
    - fbpick: `/fbpick_models` から
- pipeline 組み立て：
    - 既存 `app/static/pipeline/*` を読み込んで、graph→spec を利用
    - pipeline の Add Step に `fbpick` を追加（後述）
- pick 設定：
    - method: expectation/argmax
    - subsample on/off
    - sigma_ms_max（数値、空なら None）
    - snap on/off + feature + refine + window_ms
- 実行：
    - `Start` ボタン → `POST /batch/apply`
    - progress 表示（poll `GET /batch/job/{id}/status`）
    - 完了時に `GET /batch/job/{id}/files` で成果物リンクを出す（download URLを並べる）

### 10.2 新規 JS

**ファイル**: `app/static/batch_apply.js`（新規）

- 起動時：
    - URL query から `file_id/key1_byte/key2_byte` を取得
    - `/denoise_models`, `/fbpick_models` を fetch して dropdown 構築（label は path 表示）
- Start:
    - pipelineUI の `graphToSpec()` で `pipeline_spec` を作る
    - denoise/fbpick step の `params.model_id` を dropdown 選択値で上書き（UIの決定を spec に反映）
    - `POST /batch/apply`
- Poll:
    - `GET /batch/job/{job_id}/status`（progress は仕様通り processed/total）
- Done:
    - `GET /batch/job/{job_id}/files` → download link を出す

### 10.3 既存 pipeline UI への fbpick 組み込み

**ファイル**: `app/static/index.html`（変更）

- Add Step メニューに 1行追加：
    - `<button type="button" data-step="fbpick">FBPick</button>`

**ファイル**: `app/static/pipeline/index.js`（変更）

- `VALID_STEP_NAMES = new Set(['bandpass','denoise','fbpick'])`
- `createStep('fbpick')` は `kind:'analyzer'` で作る
- `defaultParamsFor('fbpick')` を追加（例：`tile_h=128,tile_w=6016,overlap=32,amp=true,model_id=default`）
- `PARAM_DEFS.fbpick` を追加
    - `tile_h/tile_w/overlap`（number）
    - `amp`（select: true/false）
    - `model_id`（select: `/fbpick_models` から生成した options）
- `PARAM_DEFS.denoise` に `model_id`（select: `/denoise_models` から生成）を追加
    - options label は “パス文字列”にする（dropdown+パス表示の要件を満たす）

**ファイル**: `app/static/pipeline/render/inspector.js`（変更は最小）

- 既存の `select` サポートのままで足りる（options を JS 側で埋める運用にする）
- boolean は select（`"true"/"false"`）で受けて、送信時に文字列になるので、`pipeline/index.js` 側で `parser` を仕込むか、バックエンドで bool coercion（`amp = bool(...)`）する

---

## 11) どの既存ファイルをどう触るか（一覧）

### 追加（新規ファイル）

- `app/api/routers/batch_apply.py`
- `app/services/batch_apply_service.py`
- `app/services/pipeline_spec_normalize.py`
- `app/services/fbpick_predict_service.py`
- `app/utils/denoise_models.py`
- `app/utils/gpu_lock.py`
- `app/utils/snap.py`
- `app/static/batch_apply.html`
- `app/static/batch_apply.js`

### 変更（既存ファイル）

- `app/main.py`
    - `/batch` ページ追加
    - `batch_apply_router` を include
- `app/api/routers/__init__.py`
    - `batch_apply_router` を export
- `app/api/schemas.py`
    - BatchApplyRequest/Response, PickOptions/SnapOptions 等を追加
- `app/api/routers/pipeline.py`
    - `normalize_pipeline_spec_for_cache()` を通してから `pipeline_key()` を計算＆実行
- `app/utils/denoise.py`
    - model_path 指定・モデルキャッシュを複数対応
- `app/utils/ops.py`
    - `GPU_INFER_LOCK` を適用
    - `op_denoise` が `params.model_id` を読んで解決する
- `app/api/routers/fbpick.py`（任意だが推奨）
    - `/fbpick_models` のレスポンスに `path` を含める（UIの “パス表示” 用）
- `app/static/index.html`
    - Pipeline Add Step に fbpick 追加
    - （任意）Batch Apply へのリンクボタン追加
- `app/static/pipeline/index.js`
    - fbpick step & model dropdown 対応
- `app/static/pipeline/render/inspector.js`（必要なら）
    - select/booleanの扱いを微調整

## PR1: バッチ適用の“足場”だけ（API + job管理 + artifact I/O）

**目的**

- バッチ処理の枠（開始・進捗・成果物一覧・DL）を先に固める
- 推論や巨大配列はまだやらない（最小でテストを通す）

**内容**

- 新 router: `POST /batch/apply`, `GET /batch/job/{id}/status`, `GET /batch/job/{id}/files`, `GET /batch/job/{id}/download`
- 新 schema: `BatchApplyRequest/Response`, `BatchJobStatusResponse`（最低限でOK）
- job_id 配下に `job_meta.json` だけ作る（この段階では空に近くてOK）
- `AppState.jobs` の使い方は `/pipeline/all` と同系統にする

**主な変更ファイル**

- 追加: `app/api/routers/batch_apply.py`
- 追加: `app/services/batch_apply_service.py`（start + run の空実装）
- 変更: `app/api/routers/__init__.py`, `app/main.py`（include）
- 変更: `app/api/schemas.py`（Batch系を追加）
- 変更: `app/services/pipeline_artifacts.py`（files一覧/安全DLが必要ならここに小さく足す）

**テスト**

- `app/tests/test_batch_apply_job_api.py`（新規）
    - job作成→statusが queued/running/done に遷移する（スレッドは monkeypatch で同期化すると安定）
    - download が path traversal できないこと（`name=../x` が 4xx）

**依存**

- なし（最初にマージしやすい）

---

## PR2: fbpick “pick計算”ロジックのサービス化（既存APIは挙動不変）

**目的**

- バッチ側でも使う「argmax/expectation + sigma」計算を共通化
- 既存 `POST /fbpick_predict` の計算を移植して、重複を消す（挙動は変えない）

**内容**

- `_chunked_expectations`, `_compute_picks` 相当を `app/services/fbpick_predict_math.py` などへ移す
- `fbpick_predict.py` は新サービスを呼ぶだけにする

**主な変更ファイル**

- 追加: `app/services/fbpick_predict_math.py`
- 変更: `app/api/routers/fbpick_predict.py`

**テスト**

- 既存 `test_fbpick_predict.py` がそのまま通ること（＝挙動不変の保証）
- サービス単体のユニットテストを足すなら `test_fbpick_predict_math.py`（軽量）

**依存**

- PR1不要（独立）。ただし後続PR（バッチpick保存）が楽になるので早め推奨。

---

## PR3: Snap/Subsample のPython実装（JS仕様の移植を単独PRに切る）

**目的**

- リスク高い “スナップ挙動” を単独で固める（後でバッチに組み込む）
- 「pickがNaNならsnapしない」もここで確実に

**内容**

- `viewer_legacy.js` の `adjustPickToFeature()` 相当をPythonへ移植
    - mode: peak / trough / rise（※UIでは “rise” 表記）
    - refine: parabolic / zc
    - window_ms → window_samples（dtで換算）
- ここでは“バッチ統合はしない”か、しても小さいフックだけにする

**主な変更ファイル**

- 追加: `app/utils/snap.py`（または `app/services/snap.py`）

**テスト**

- `app/tests/test_snap.py`（新規）
    - peak/trough/riseで期待通り近傍に吸着する（小配列で決定的に）
    - refine(parabolic/zc) が範囲外に飛ばない
    - NaN入力はそのまま返す（バッチ側の条件とも整合）

**依存**

- なし（PR2とも独立）。ただし PR5（予測pick保存）で使う。

---

## PR4: バッチ本体（3D npy生成まで：denoise/prob + padding + key配列）

**目的**

- “全key1一括” と “padding付き3D npy保存” をまず完成させる
- pick(npz) はまだ入れない（次PRへ）

**内容**

- job runner が `TraceStoreSectionReader.get_key1_values()`（= `np.unique(key1s)`）で全セクション処理
- `max_traces` を事前走査で確定
- 3D出力（memmap推奨）
    - `denoise_f32_padded.npy`（float32）
    - `fbpick_prob_f16_padded.npy`（float16）
    - `key1_values.npy`（int32）
    - `key2_values_padded.npy`（int32、padding=0）
- paddingはすべて 0 埋め
- 進捗は `processed_key1/total_key1`
- 1セクション例外で全停止（job status=error）
- GPU同時推論禁止（このPRか次PRで lock を入れる。入れるならこのPRが筋が良い）

**主な変更ファイル**

- 追加/拡張: `app/services/batch_apply_service.py`（実処理実装）
- 追加: `app/utils/gpu_lock.py`（入れるなら）
- 変更: `app/utils/ops.py`（denoise/fbpick を lock で囲むなら）
- 変更: `app/api/schemas.py`（BatchApplyRequest を拡張：outputs/pipeline_spec など）

**テスト**

- `app/tests/test_batch_apply_artifacts.py`（新規）
    - monkeypatchで `apply_pipeline` や `op_fbpick/op_denoise` を軽量化し、形だけ検証
    - `key1` 順序が `np.unique` と一致
    - `key2_values_padded` の padding が 0
    - 3D npy の dtype/shape が仕様通り
    - downsample が存在しないこと（出力shapeが元サンプル数のまま）

**依存**

- PR1（API足場）が先にあると楽
- PR2/3は不要（この段階はpick保存しないため）

---

## PR5: 予測pick保存（manual npz互換 + sigma gate + snap/subsample）

**目的**

- `predicted_picks_time_s.npz` を “manual export互換” で保存
- method(expectation/argmax), subsample, sigma_ms_max, snap を入れる
- “pickがNaNならsnapしない” を満たす

**内容**

- PR2の共通pick計算を利用（expectation/argmax/sigma）
- sigma gate：`sigma_ms > sigma_ms_max` は NaN
- `sorted_to_original` 変換は `picks.py` と同様のルールで実装
- 保存キー/型は `export_manual_picks_npz()` と一致（最低限）
    - `picks_time_s`, `n_traces`, `n_samples`, `dt`, `format_version(=1)`, `exported_at`, `export_app`, `source_hint`
- 追加情報（任意）は別キーで入れてもOK（読めなくなる危険があるなら `job_meta.json` 側に寄せるのが安全）

**主な変更ファイル**

- 変更: `app/services/batch_apply_service.py`（npz保存の追加）
- （使うなら）変更: `app/services/fbpick_predict_math.py`（subsample対応など）
- 変更: `app/api/schemas.py`（PickOptions/SnapOptions を確定）

**テスト**

- `app/tests/test_batch_apply_picks_npz.py`（新規）
    - npz の必須キーが揃う/型が一致
    - sigma gate で NaN になる
    - NaN pick は snap されない（snap関数を monkeypatch して呼ばれないことを検証）
    - `sorted_to_original` の並びが `export_manual_picks_npz` と同じ規約

**依存**

- PR2（pick計算共通化）
- PR3（snap Python実装）
- PR4（バッチ本体）

---

## PR6: UI（別画面/別window：バッチ実行・モデル選択・成果物DL）

**目的**

- 要件の「選択が多いので別画面/別window」を実現
- まずは “実行→進捗→成果物DL” を完成（巨大3Dの可視化は次段でも可）

**内容**

- `/batch` ページ追加（`app/static/batch_apply.html` + `batch_apply.js`）
- 選択UI
    - 処理対象（denoise / fbpick / pipeline）
    - primary/secondary key
    - 重みパス dropdown（表示としてパス）
    - pick設定（method/sigma/snap mode/refine/window）
    - denoise parameter
    - pipeline editor は既存の pipeline UI を流用（可能なら）し、fbpick step も追加
- 実行ボタン→`POST /batch/apply`→status polling→files→downloadリンク

**主な変更ファイル**

- 追加: `app/static/batch_apply.html`, `app/static/batch_apply.js`
- 変更: `app/main.py`（`/batch` route）
- 変更: `app/static/index.html`（Batch画面を開く導線ボタンを追加するなら）
- 変更: `app/static/pipeline/index.js`（fbpick step 追加・モデルdropdown対応をやるならここで）

**テスト**

- UIは最初はE2Eなしでも良いが、最低限：
    - JSユニット（`vitest`）で request payload の組み立てを1本だけでも入れると壊れにくい

**依存**

- PR1（API）、PR4/5（成果物が出る）
