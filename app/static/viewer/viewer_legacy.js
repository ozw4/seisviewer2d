



    let __dbg = { enabled: true };
    // 連続重複を抑制したいタグ（必要なら増やせる）
      const DEDUP_TAGS = new Set(['RENDER@picks']);
    const __lastLogKeyByTag = new Map();
    function __stableStringify(v) {
        if (v && typeof v === 'object') {
            if (Array.isArray(v)) return `[${v.map(__stableStringify).join(',')}]`;
            const keys = Object.keys(v).sort();
            return `{${keys.map(k => JSON.stringify(k) + ':' + __stableStringify(v[k])).join(',')}}`;
          }
        return JSON.stringify(v);
      }
    function D(tag, obj) {
      if (!__dbg.enabled) return;
      // 直前と同じ内容ならスキップ（指定タグのみ）
        if (DEDUP_TAGS.has(tag)) {
            let key;
            try { key = __stableStringify(obj); } catch { key = null; }
            const last = __lastLogKeyByTag.get(tag);
            if (last === key) return;
            __lastLogKeyByTag.set(tag, key);
          }
      try {
          console.debug(`🧭 ${tag}`, JSON.parse(JSON.stringify(obj)));
        } catch {
            console.debug(`🧭 ${tag}`, obj);
          }
    }


    function samplePicks(arr, k = 6) {
      return (arr || []).slice(0, k).map(p => ({ tr: Math.round(p.trace), t: +p.time }));
    }
    // 現在の可視Xレンジ（整数トレース）と viewer 状態サマリ
    function visibleXRng() {
      const gd = document.getElementById('plot');
      const xr = gd?._fullLayout?.xaxis?.range || savedXRange || [renderedStart ?? 0, renderedEnd ?? 0];
      const x0 = Math.floor(Math.min(xr[0], xr[1]));
      const x1 = Math.ceil(Math.max(xr[0], xr[1]));
      return [x0, x1];
    }
    function viewerState(tag = 'STATE') {
      const [x0, x1] = visibleXRng();
      const mode = latestWindowRender?.mode || 'none';
      const st = {
        key1: key1Values?.[parseInt(document.getElementById('key1_slider')?.value || '0', 10)],
        layer: document.getElementById('layerSelect')?.value,
        mode, stepX: latestWindowRender?.stepX, stepY: latestWindowRender?.stepY,
        x0, x1, renderedStart, renderedEnd, savedXRange, picksCount: picks?.length || 0
      };
      D(tag, st);
      return st;
    }
    // ローカル→サーバ疑い時に呼べる即席ダンプ

    var currentScaling = 'amax';
    var sectionShape = null;
    var renderedStart = null;
    var renderedEnd = null;
    var picks = [];
    var predictedPicks = [];
    const fbPredCache = new Map(); // key: "key1|layer|pipelineKey|modelId"
    const FBPICK_MODEL_STORAGE_KEY = 'fbpick_model_id';
    var currentFbKey = null;
    var fbPredReqId = 0;
    var downsampleFactor = 1;
    var isPickMode = false;
    var linePickStart = null;
    var deleteRangeStart = null;
    let manualPickUndoStack = [];

    function formatPendingPickTime(time) {
      return Number(time).toFixed(3);
    }

    function getPendingPickOverlayState() {
      if (linePickStart && Number.isFinite(linePickStart.trace) && Number.isFinite(linePickStart.time)) {
        return {
          kind: 'line',
          trace: linePickStart.trace | 0,
          time: Number(linePickStart.time),
        };
      }
      const deleteAnchor = normalizeDeleteRangeAnchor(deleteRangeStart);
      if (deleteAnchor) {
        return {
          kind: 'delete-range',
          trace: deleteAnchor.trace,
        };
      }
      return null;
    }

    function hasPendingPickOverlayState() {
      return getPendingPickOverlayState() !== null;
    }

    function renderPendingPickStatus() {
      const node = document.getElementById('pendingPickStatus');
      if (!node) return;
      const pending = getPendingPickOverlayState();
      if (!pending) {
        node.textContent = '';
        node.hidden = true;
        return;
      }
      if (pending.kind === 'line') {
        node.textContent = `Line pick anchor: trace ${pending.trace}, time ${formatPendingPickTime(pending.time)} s`;
      } else {
        node.textContent = `Delete range anchor: trace ${pending.trace}`;
      }
      node.hidden = false;
    }

    function syncPendingPickUi() {
      renderPendingPickStatus();
      schedulePickOverlayUpdate();
    }

    function clearPendingPickState(reason = '', options = {}) {
      const keepLine = options.keepLine === true;
      const keepDeleteRange = options.keepDeleteRange === true;
      const prevLine = !!linePickStart;
      const prevDelete = deleteRangeStart !== null;
      if (!keepLine) linePickStart = null;
      if (!keepDeleteRange) deleteRangeStart = null;
      if (prevLine || prevDelete || reason) {
        D('PICKS@pending:clear', { reason, keepLine, keepDeleteRange });
      }
      syncPendingPickUi();
    }

    function setLinePickAnchor(trace, time) {
      linePickStart = { trace: trace | 0, time: Number(time) };
      deleteRangeStart = null;
      syncPendingPickUi();
    }

    function normalizeDeleteRangeAnchor(anchor) {
      if (anchor === null || anchor === undefined) return null;
      if (typeof anchor === 'object') {
        const trace = Number(anchor.trace);
        const time = Number(anchor.time);
        if (!Number.isFinite(trace)) return null;
        return {
          trace: trace | 0,
          time: Number.isFinite(time) ? time : NaN,
        };
      }
      const trace = Number(anchor);
      if (!Number.isFinite(trace)) return null;
      return { trace: trace | 0, time: NaN };
    }

    function setDeleteRangeAnchor(trace, time) {
      deleteRangeStart = { trace: trace | 0, time: Number(time) };
      linePickStart = null;
      syncPendingPickUi();
    }

    window.getPendingPickOverlayState = getPendingPickOverlayState;
    window.hasPendingPickOverlayState = hasPendingPickOverlayState;
    let manualPickRedoStack = [];
    let applyingManualPickHistory = false;

    let uiResetNonce = 0;
    function currentUiRevision() {
      const sel = document.getElementById('layerSelect');
      const layer = sel ? sel.value : 'raw';
      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];
      const pKey = window.latestPipelineKey || '';
      return `rev:${currentFileId}|${key1Val}|${layer}|${pKey}|${uiResetNonce}`;
    }

    const OP_STATUS_IDS = {
      export: 'op-status-export',
      import: 'op-status-import',
      predict: 'op-status-predict',
    };
    const OP_STATUS_CLASSES = ['is-idle', 'is-running', 'is-success', 'is-error'];
    const OP_STATE_VERSION = {
      export: 0,
      import: 0,
      predict: 0,
    };
    const OP_TOAST_SUCCESS_MS = 3500;
    let toastSeq = 0;
    const toastByKey = new Map();
    const toastTimers = new Map();

    function opStatusNode(op) {
      const nodeId = OP_STATUS_IDS[op];
      if (!nodeId) return null;
      return document.getElementById(nodeId);
    }

    function setOpStatus(op, state, message = '') {
      const node = opStatusNode(op);
      if (!node) return;
      const nextState = OP_STATUS_IDS[op] && OP_STATUS_CLASSES.includes(`is-${state}`)
        ? state
        : 'idle';
      for (const className of OP_STATUS_CLASSES) {
        node.classList.remove(className);
      }
      node.classList.add(`is-${nextState}`);
      const valueNode = node.querySelector('.op-value');
      if (valueNode) {
        valueNode.textContent = message || nextState;
      }
    }

    function beginOpStatus(op, message = 'running') {
      if (!(op in OP_STATE_VERSION)) return 0;
      OP_STATE_VERSION[op] += 1;
      const token = OP_STATE_VERSION[op];
      setOpStatus(op, 'running', message);
      return token;
    }

    function setOpStatusIfCurrent(op, token, state, message = '') {
      if (!(op in OP_STATE_VERSION)) return false;
      if (OP_STATE_VERSION[op] !== token) return false;
      setOpStatus(op, state, message);
      return true;
    }

    function resetOpStatuses() {
      for (const op of Object.keys(OP_STATUS_IDS)) {
        OP_STATE_VERSION[op] += 1;
        setOpStatus(op, 'idle', 'idle');
      }
    }

    const VIEWER_EMPTY_STATE_COPY = {
      'no-dataset': {
        title: 'No dataset open',
        description: 'Open or upload a SEG-Y file to start exploring sections.',
        helper: 'After opening a dataset, the viewer will show the first available section.',
      },
      unavailable: {
        title: 'Dataset unavailable',
        description: 'The previously selected dataset could not be opened. Open or upload a SEG-Y file to continue.',
        helper: 'After opening a dataset, the viewer will show the first available section.',
      },
    };

    function getViewerEmptyStateNodes() {
      return {
        root: document.getElementById('viewerEmptyState'),
        title: document.getElementById('viewerEmptyStateTitle'),
        description: document.getElementById('viewerEmptyStateDescription'),
        helper: document.getElementById('viewerEmptyStateHelper'),
      };
    }

    function clearPlotSurface() {
      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;
      if (typeof window.Plotly?.purge === 'function') {
        try { window.Plotly.purge(plotDiv); } catch (_) { }
      }
      plotDiv.innerHTML = '';
      plotDiv.classList.remove('js-plotly-plot');
    }

    function showViewerEmptyState(kind = 'no-dataset') {
      const copy = VIEWER_EMPTY_STATE_COPY[kind] || VIEWER_EMPTY_STATE_COPY['no-dataset'];
      const { root, title, description, helper } = getViewerEmptyStateNodes();
      if (title) title.textContent = copy.title;
      if (description) description.textContent = copy.description;
      if (helper) helper.textContent = copy.helper;
      if (typeof hideLoading === 'function') hideLoading();
      clearPlotSurface();
      if (root) {
        root.dataset.state = kind;
        root.hidden = false;
      }
      updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
    }

    function hideViewerEmptyState() {
      const { root } = getViewerEmptyStateNodes();
      if (!root) return;
      root.hidden = true;
      delete root.dataset.state;
    }

    function isViewerEmptyStateVisible() {
      const { root } = getViewerEmptyStateNodes();
      return !!root && root.hidden === false;
    }

    function getSectionNavNodes() {
      return {
        slider: document.getElementById('key1_slider'),
        display: document.getElementById('key1_val_display'),
        prev: document.getElementById('sectionNavPrev'),
        next: document.getElementById('sectionNavNext'),
        position: document.getElementById('sectionNavPosition'),
        key1Current: document.getElementById('sectionNavKey1Current'),
        jumpInput: document.getElementById('sectionNavKey1Input'),
        jumpGo: document.getElementById('sectionNavKey1Go'),
        validation: document.getElementById('sectionNavValidation'),
      };
    }

    function setSectionNavValidation(message = '') {
      const { validation } = getSectionNavNodes();
      if (!validation) return;
      const text = typeof message === 'string' ? message.trim() : '';
      validation.textContent = text;
      validation.hidden = text.length === 0;
    }

    function clearSectionNavValidation() {
      setSectionNavValidation('');
    }

    function resolveKey1Value(rawValue) {
      if (!Array.isArray(key1Values) || key1Values.length === 0) {
        return { ok: false, message: 'No sections are available.' };
      }
      const text = String(rawValue ?? '').trim();
      if (!text) {
        return { ok: false, message: 'Enter a key1 value.' };
      }
      const value = Number(text);
      if (!Number.isFinite(value) || !Number.isInteger(value)) {
        return { ok: false, message: 'Enter a valid integer key1 value.' };
      }
      const index = key1Values.indexOf(value);
      if (index < 0) {
        return { ok: false, message: `key1 ${value} is not available in this dataset.` };
      }
      return { ok: true, index, value };
    }

    function getCurrentKey1Index() {
      const { slider } = getSectionNavNodes();
      if (!slider || !Array.isArray(key1Values) || key1Values.length === 0) return -1;
      let idx = parseInt(slider.value, 10);
      if (!Number.isFinite(idx)) idx = 0;
      idx = Math.max(0, Math.min(key1Values.length - 1, idx));
      if (slider.value !== String(idx)) {
        slider.value = String(idx);
      }
      return idx;
    }

    function updateSectionNavigation(options = {}) {
      const syncDisplay = options.syncDisplay !== false;
      const syncJumpInput = options.syncJumpInput === true;
      const nodes = getSectionNavNodes();
      const hasSections = Array.isArray(key1Values) && key1Values.length > 0;
      const disabled = !currentFileId || !hasSections;

      let idx = -1;
      if (nodes.slider) {
        nodes.slider.min = '0';
        nodes.slider.max = hasSections ? String(key1Values.length - 1) : '0';
        if (!hasSections) {
          nodes.slider.value = '0';
        }
        nodes.slider.disabled = disabled;
      }
      if (hasSections) {
        idx = getCurrentKey1Index();
      }

      if (nodes.display) {
        nodes.display.disabled = false;
        if (hasSections) {
          nodes.display.min = String(key1Values[0]);
          nodes.display.max = String(key1Values[key1Values.length - 1]);
          if (syncDisplay && idx >= 0) {
            nodes.display.value = String(key1Values[idx]);
          }
        } else {
          nodes.display.removeAttribute('min');
          nodes.display.removeAttribute('max');
          if (syncDisplay) {
            nodes.display.value = '';
          }
        }
      }

      if (nodes.prev) {
        nodes.prev.disabled = disabled || idx <= 0;
      }
      if (nodes.next) {
        nodes.next.disabled = disabled || idx < 0 || idx >= key1Values.length - 1;
      }
      if (nodes.position) {
        nodes.position.textContent = hasSections && idx >= 0
          ? `Section ${idx + 1} / ${key1Values.length}`
          : 'Section - / -';
      }
      if (nodes.key1Current) {
        nodes.key1Current.textContent = hasSections && idx >= 0
          ? `key1: ${key1Values[idx]}`
          : 'key1: -';
      }
      if (nodes.jumpInput) {
        nodes.jumpInput.disabled = disabled;
        if (!disabled) {
          nodes.jumpInput.min = String(key1Values[0]);
          nodes.jumpInput.max = String(key1Values[key1Values.length - 1]);
          if (syncJumpInput && idx >= 0) {
            nodes.jumpInput.value = String(key1Values[idx]);
          }
        } else {
          nodes.jumpInput.removeAttribute('min');
          nodes.jumpInput.removeAttribute('max');
          if (syncJumpInput) {
            nodes.jumpInput.value = '';
          }
        }
      }
      if (nodes.jumpGo) {
        nodes.jumpGo.disabled = disabled;
      }
      if (disabled) {
        clearSectionNavValidation();
      }
    }

    function selectKey1Index(index, options = {}) {
      const { slider } = getSectionNavNodes();
      if (!slider || !Array.isArray(key1Values) || key1Values.length === 0) return false;
      const prevIndex = getCurrentKey1Index();
      let nextIndex = Number(index);
      if (!Number.isFinite(nextIndex)) return false;
      nextIndex = Math.trunc(nextIndex);
      nextIndex = Math.max(0, Math.min(key1Values.length - 1, nextIndex));
      slider.value = String(nextIndex);
      updateSectionNavigation({
        syncDisplay: true,
        syncJumpInput: options.syncJumpInput === true,
      });
      return nextIndex !== prevIndex;
    }

    function removeToast(toastNode) {
      if (!toastNode) return;
      const toastId = toastNode.dataset.toastId || '';
      if (toastTimers.has(toastId)) {
        clearTimeout(toastTimers.get(toastId));
        toastTimers.delete(toastId);
      }
      const dedupeKey = toastNode.dataset.toastKey || '';
      if (dedupeKey) {
        const mapped = toastByKey.get(dedupeKey);
        if (mapped === toastNode) {
          toastByKey.delete(dedupeKey);
        }
      }
      toastNode.remove();
    }

    function pushToast({ level = 'info', title = '', message = '', sticky = false, dedupeKey = '' }) {
      const host = document.getElementById('opToastHost');
      if (!host) return null;

      if (dedupeKey) {
        const prev = toastByKey.get(dedupeKey);
        if (prev) {
          removeToast(prev);
        }
      }

      const nextLevel = ['info', 'success', 'error'].includes(level) ? level : 'info';
      const toast = document.createElement('div');
      const toastId = String(++toastSeq);
      toast.className = `toast toast-${nextLevel}`;
      toast.dataset.toastId = toastId;
      if (dedupeKey) {
        toast.dataset.toastKey = dedupeKey;
      }
      toast.setAttribute('role', nextLevel === 'error' ? 'alert' : 'status');
      toast.setAttribute('aria-live', nextLevel === 'error' ? 'assertive' : 'polite');

      const head = document.createElement('div');
      head.className = 'toast-head';
      const titleNode = document.createElement('span');
      titleNode.className = 'toast-title';
      titleNode.textContent = title || 'Notice';
      const closeBtn = document.createElement('button');
      closeBtn.className = 'toast-close';
      closeBtn.type = 'button';
      closeBtn.setAttribute('aria-label', 'Close notification');
      closeBtn.textContent = 'x';
      closeBtn.addEventListener('click', () => removeToast(toast));
      head.appendChild(titleNode);
      head.appendChild(closeBtn);

      const messageNode = document.createElement('div');
      messageNode.className = 'toast-message';
      messageNode.textContent = message || '';

      toast.appendChild(head);
      toast.appendChild(messageNode);
      host.appendChild(toast);
      if (dedupeKey) {
        toastByKey.set(dedupeKey, toast);
      }

      if (!sticky) {
        const timerId = setTimeout(() => removeToast(toast), OP_TOAST_SUCCESS_MS);
        toastTimers.set(toastId, timerId);
      }
      return toast;
    }

    async function readErrorDetail(response) {
      if (!response) return '';
      try {
        const contentType = response.headers?.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const payload = await response.json();
          if (payload && typeof payload.detail === 'string') {
            return payload.detail;
          }
          if (payload && payload.detail != null) {
            return String(payload.detail);
          }
          return payload ? JSON.stringify(payload) : '';
        }
        const text = await response.text();
        if (!text) return '';
        try {
          const parsed = JSON.parse(text);
          if (parsed && typeof parsed.detail === 'string') {
            return parsed.detail;
          }
        } catch (_) {
          // plain-text response
        }
        return text;
      } catch (_) {
        return '';
      }
    }


    // --- 追加：ハンドラ ---
    function onKey1Input() {
      updateKey1Display();
      clearSectionNavValidation();
      if (!currentFileId || !Array.isArray(key1Values) || key1Values.length === 0) return;
      fetchAndPlotDebounced();        // 入力が止まってから実行
    }

    async function onKey1Change(options = {}) {
      const immediate = options.immediate !== false;
      updateKey1Display();
      clearSectionNavValidation();
      const slider = document.getElementById('key1_slider');
      if (slider) {
          slider.blur();
        }
      if (!currentFileId || !Array.isArray(key1Values) || key1Values.length === 0) return;
      const debounced = ensureFlushPickOpsDebounced();
      if (typeof debounced?.flush === 'function') {
        await debounced.flush();
      } else {
        await flushPickOps();
      }
      if (immediate) {
        fetchAndPlotDebounced.flush();
      } else {
        fetchAndPlotDebounced();
      }
    }

    function onKey1ValueDisplayInput() {
      clearSectionNavValidation();
    }

    function syncSliderWithInput() {
      const { display } = getSectionNavNodes();
      if (!display) return false;
      const resolved = resolveKey1Value(display.value);
      if (!resolved.ok) {
        setSectionNavValidation(resolved.message);
        updateSectionNavigation({ syncDisplay: true });
        return false;
      }
      clearSectionNavValidation();
      selectKey1Index(resolved.index);
      return true;
    }

    function onKey1ValueDisplayChange() {
      if (!syncSliderWithInput()) return;
      onKey1Change({ immediate: false }).catch((err) => console.warn('key1 input change failed', err));
    }

    function onSectionNavJumpInput() {
      clearSectionNavValidation();
    }

    function onSectionNavJumpKeyDown(event) {
      if (!event || event.key !== 'Enter') return;
      event.preventDefault();
      goToSectionByKey1();
    }

    function goToSectionByKey1() {
      const { jumpInput } = getSectionNavNodes();
      if (!jumpInput) return;
      const resolved = resolveKey1Value(jumpInput.value);
      if (!resolved.ok) {
        setSectionNavValidation(resolved.message);
        return;
      }
      clearSectionNavValidation();
      selectKey1Index(resolved.index, { syncJumpInput: true });
      onKey1Change({ immediate: true }).catch((err) => console.warn('key1 jump failed', err));
    }

    function goToPreviousSection() {
      changeSectionByDelta(-1);
    }

    function goToNextSection() {
      changeSectionByDelta(1);
    }



    function snapshotAxesRangesFromDOM() {
      const gd = document.getElementById('plot');
      const xa = gd?._fullLayout?.xaxis;
      const ya = gd?._fullLayout?.yaxis;
      if (xa && Array.isArray(xa.range) && xa.range.length === 2) {
        savedXRange = [xa.range[0], xa.range[1]];
      }
      if (ya && Array.isArray(ya.range) && ya.range.length === 2) {
        const y0 = ya.range[0], y1 = ya.range[1];
        // 上下逆レンジでも扱えるように max/min 順で保存
        savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
      }
    }

    function readAxisRange(ev, axisName) {
      if (!ev || typeof ev !== 'object') return null;

      const packed = ev[`${axisName}.range`];
      if (Array.isArray(packed) && packed.length === 2) {
        return [packed[0], packed[1]];
      }

      const k0 = `${axisName}.range[0]`;
      const k1 = `${axisName}.range[1]`;
      if (k0 in ev && k1 in ev) {
        return [ev[k0], ev[k1]];
      }

      return null;
    }

    function filenameFromContentDisposition(disposition) {
      if (!disposition) return null;
      const utfMatch = /filename\*=UTF-8''([^;\n]+)/i.exec(disposition);
      if (utfMatch && utfMatch[1]) {
        try {
          return decodeURIComponent(utfMatch[1]);
        } catch (err) {
          console.warn('Failed to decode UTF-8 filename from Content-Disposition:', err);
          return utfMatch[1];
        }
      }
      const match = /filename="?([^";]+)"?/i.exec(disposition);
      return match && match[1] ? match[1] : null;
    }

    function saveBlob(uint8OrBlob, filename) {
        const blob = (uint8OrBlob instanceof Blob)
          ? uint8OrBlob
          : new Blob([uint8OrBlob], { type: 'application/octet-stream' });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'download.bin';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

    async function exportManualPicksFile({
      endpoint,
      buttonId,
      fallbackFilename,
      key1Byte,
      includeKey1Byte = true,
    }) {
      const token = beginOpStatus('export', 'preparing...');
      const exportBtn = document.getElementById(buttonId);
      if (exportBtn) exportBtn.disabled = true;
      try {
        if (!currentFileId) {
          const message = 'file_id not loaded';
          if (setOpStatusIfCurrent('export', token, 'error', message)) {
            pushToast({
              level: 'error',
              title: 'Export failed',
              message,
              sticky: true,
              dedupeKey: 'export:error',
            });
          }
          return;
        }

        await flushPickOps();
        if (!setOpStatusIfCurrent('export', token, 'running', 'exporting...')) return;

        const params = new URLSearchParams({
          file_id: String(currentFileId),
          key2_byte: String(currentKey2Byte),
        });
        if (includeKey1Byte) {
          const resolvedKey1Byte = Number.isFinite(Number(key1Byte))
            ? String(parseInt(String(key1Byte), 10))
            : String(currentKey1Byte);
          params.set('key1_byte', resolvedKey1Byte);
        }
        const response = await fetch(`${endpoint}?${params.toString()}`);
        if (!response.ok) {
          const detail = await readErrorDetail(response);
          const message = detail
            ? `HTTP ${response.status}: ${detail}`
            : `HTTP ${response.status}`;
          if (setOpStatusIfCurrent('export', token, 'error', message)) {
            pushToast({
              level: 'error',
              title: 'Export failed',
              message,
              sticky: true,
              dedupeKey: 'export:error',
            });
          }
          return;
        }

        const blob = await response.blob();
        const disposition = response.headers.get('content-disposition');
        const filename = filenameFromContentDisposition(disposition)
          || fallbackFilename;
        saveBlob(blob, filename);
        if (setOpStatusIfCurrent('export', token, 'success', `saved ${filename}`)) {
          pushToast({
            level: 'success',
            title: 'Export complete',
            message: `Saved ${filename}`,
            dedupeKey: 'export:success',
          });
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (setOpStatusIfCurrent('export', token, 'error', message)) {
          pushToast({
            level: 'error',
            title: 'Export failed',
            message,
            sticky: true,
            dedupeKey: 'export:error',
          });
        }
      } finally {
        if (exportBtn) exportBtn.disabled = false;
      }
    }

    async function exportManualPicksNpz() {
      await exportManualPicksFile({
        endpoint: '/export_manual_picks_npz',
        buttonId: 'export-manual-picks-npz',
        fallbackFilename: `manual_picks_time_v1_${currentFileId}.npz`,
      });
    }

    async function exportManualPicksGrstatTxt() {
      await exportManualPicksFile({
        endpoint: '/export_manual_picks_grstat_txt',
        buttonId: 'export-manual-picks-grstat-txt',
        fallbackFilename: `manual_picks_grstat_v1_${currentFileId}.txt`,
        includeKey1Byte: false,
      });
    }

    async function importManualPicksNpz(file) {
      if (!file) {
        return;
      }
      const token = beginOpStatus('import', 'preparing...');
      const importBtn = document.getElementById('import-manual-picks-npz');
      const modeNode = document.getElementById('manual-picks-import-mode');
      const mode = modeNode && modeNode.value === 'merge' ? 'merge' : 'replace';
      if (importBtn) importBtn.disabled = true;
      if (modeNode) modeNode.disabled = true;
      try {
        if (!currentFileId) {
          const message = 'file_id not loaded';
          if (setOpStatusIfCurrent('import', token, 'error', message)) {
            pushToast({
              level: 'error',
              title: 'Import failed',
              message,
              sticky: true,
              dedupeKey: 'import:error',
            });
          }
          return;
        }

        await flushPickOps();
        if (!setOpStatusIfCurrent('import', token, 'running', 'uploading...')) return;

        const params = new URLSearchParams({
          file_id: String(currentFileId),
          key1_byte: String(currentKey1Byte),
          key2_byte: String(currentKey2Byte),
          mode,
        });
        const form = new FormData();
        form.append('file', file);
        const response = await fetch(`/import_manual_picks_npz?${params.toString()}`, {
          method: 'POST',
          body: form,
        });
        if (!response.ok) {
          const detail = await readErrorDetail(response);
          const message = detail
            ? `HTTP ${response.status}: ${detail}`
            : `HTTP ${response.status}`;
          if (setOpStatusIfCurrent('import', token, 'error', message)) {
            pushToast({
              level: 'error',
              title: 'Import failed',
              message,
              sticky: true,
              dedupeKey: 'import:error',
            });
          }
          return;
        }

        let payload = {};
        try {
          payload = await response.json();
        } catch (_) {
          payload = {};
        }

        if (!setOpStatusIfCurrent('import', token, 'running', 'reloading picks...')) return;
        await reloadPicksForCurrentSection();
        clearManualPickHistory();
        renderLatestView();
        const applied = Number.isFinite(Number(payload?.applied))
          ? Number(payload.applied)
          : null;
        const droppedNeg = Number.isFinite(Number(payload?.dropped_neg))
          ? Number(payload.dropped_neg)
          : 0;
        const clampedHi = Number.isFinite(Number(payload?.clamped_hi))
          ? Number(payload.clamped_hi)
          : 0;
        const summary = [
          `mode=${mode}`,
          applied === null ? null : `applied=${applied}`,
          `dropped=${droppedNeg}`,
          `clamped=${clampedHi}`,
        ].filter(Boolean).join(', ');
        if (setOpStatusIfCurrent('import', token, 'success', summary || 'imported')) {
          pushToast({
            level: 'success',
            title: 'Import complete',
            message: summary || 'Import completed.',
            dedupeKey: 'import:success',
          });
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (setOpStatusIfCurrent('import', token, 'error', message)) {
          pushToast({
            level: 'error',
            title: 'Import failed',
            message,
            sticky: true,
            dedupeKey: 'import:error',
          });
        }
      } finally {
        if (importBtn) importBtn.disabled = false;
        if (modeNode) modeNode.disabled = false;
      }
    }

    function triggerManualPicksImportNpz() {
      const node = document.getElementById('manual-picks-import-input');
      if (!node) {
        return;
      }
      node.value = '';
      node.click();
    }

    async function onManualPicksImportInputChange(ev) {
      const node = ev && ev.target ? ev.target : null;
      const file = node && node.files && node.files[0] ? node.files[0] : null;
      await importManualPicksNpz(file);
      if (node) {
        node.value = '';
      }
    }

    let suppressRelayout = false;       // ignore relayouts we cause internally
    let isRelayouting = false;          // true while user is actively adjusting viewport
    let pendingResetFetch = false;      // reset event arrived while dragging; run once after settle
    let lastImmediateWindowFetchAt = 0; // immediate-reset fetch dedup
    const RESET_FETCH_DEDUP_MS = 100;
    let forceFullExtentOnce = false;    // next window calc uses full extent with no padding
    let pickOverlayRaf = 0;
    let pickOverlayDirty = false;
    let pendingHotkeyXPanDelta = 0;
    let hotkeyXPanRaf = 0;
    let plotHover = false;

    // 追加：現在のFB計算に紐づくレイヤ/パイプラインキー
    let currentFbLayer = 'raw';
    let currentFbPipelineKey = null;

    function currentManualPickHistorySectionKey() {
      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values?.[idx];
      return `${currentFileId}|${currentKey1Byte}|${currentKey2Byte}|${key1Val ?? 'unknown'}`;
    }

    function cloneManualPick(pick) {
      if (!pick || !Number.isFinite(pick.trace) || !Number.isFinite(pick.time)) {
        return null;
      }
      return { trace: (pick.trace | 0), time: +pick.time };
    }

    function getLocalPickOnTrace(traceInt) {
      const found = picks.find((p) => (p.trace | 0) === traceInt);
      return cloneManualPick(found);
    }

    function localPickDisplayTime(pick) {
      const trace = Number(pick?.trace);
      const rawTime = Number(pick?.time);
      if (!Number.isFinite(trace) || !Number.isFinite(rawTime)) return NaN;
      const converter = window.pickRawTimeToDisplayTime || window.rawTimeToDisplayTime;
      if (typeof converter !== 'function') return NaN;
      return Number(converter(trace, rawTime, pick));
    }

    function getLocalPickOnTraceForDisplayClick(traceInt, displayTime) {
      const targetTime = Number(displayTime);
      if (!Number.isFinite(targetTime)) return null;

      let best = null;
      let bestDistance = Infinity;
      for (const pick of picks) {
        if ((pick?.trace | 0) !== traceInt) continue;
        const pickDisplayTime = localPickDisplayTime(pick);
        if (!Number.isFinite(pickDisplayTime)) continue;
        const distance = Math.abs(targetTime - pickDisplayTime);
        if (distance < bestDistance) {
          best = pick;
          bestDistance = distance;
        }
      }
      return best ? cloneManualPick(best) : null;
    }

    function getLocalPicksInTraceRangeForDisplayDelete(anchor, trace, displayTime) {
      const startAnchor = normalizeDeleteRangeAnchor(anchor);
      if (!startAnchor) return [];

      const x0 = startAnchor.trace;
      const x1 = trace | 0;
      const y0 = Number(startAnchor.time);
      const y1 = Number(displayTime);
      const start = Math.min(x0, x1);
      const end = Math.max(x0, x1);
      const canInterpolateTime = Number.isFinite(y0) && Number.isFinite(y1);
      const slope = canInterpolateTime && x1 !== x0 ? (y1 - y0) / (x1 - x0) : 0;
      const nearestByTrace = new Map();

      for (const pick of picks) {
        const traceInt = pick?.trace | 0;
        if (traceInt < start || traceInt > end) continue;

        const pickDisplayTime = localPickDisplayTime(pick);
        if (!Number.isFinite(pickDisplayTime)) continue;

        const targetDisplayTime = canInterpolateTime
          ? (x1 === x0 ? y1 : y0 + slope * (traceInt - x0))
          : pickDisplayTime;
        const distance = Math.abs(targetDisplayTime - pickDisplayTime);
        const current = nearestByTrace.get(traceInt);
        if (!current || distance < current.distance) {
          nearestByTrace.set(traceInt, { pick, distance });
        }
      }

      return Array.from(nearestByTrace.values())
        .map(({ pick }) => cloneManualPick(pick))
        .filter(Boolean)
        .sort((a, b) => a.trace - b.trace);
    }

    function manualPickStateEquals(a, b) {
      if (a == null && b == null) return true;
      if (!a || !b) return false;
      return (a.trace | 0) === (b.trace | 0) && +a.time === +b.time;
    }

    function updateManualPickHistoryButtons() {
      const undoBtn = document.getElementById('manual-picks-undo');
      const redoBtn = document.getElementById('manual-picks-redo');
      if (undoBtn) undoBtn.disabled = manualPickUndoStack.length === 0;
      if (redoBtn) redoBtn.disabled = manualPickRedoStack.length === 0;
    }

    function clearManualPickHistory() {
      manualPickUndoStack = [];
      manualPickRedoStack = [];
      clearPendingPickState('manual-history-clear');
      updateManualPickHistoryButtons();
    }

    function pushManualPickHistoryEntry(entry) {
      if (!entry || !Array.isArray(entry.changes) || entry.changes.length === 0) {
        updateManualPickHistoryButtons();
        return;
      }
      manualPickUndoStack.push(entry);
      manualPickRedoStack = [];
      updateManualPickHistoryButtons();
    }

    function applyManualPickState(traceInt, pickState) {
      removeAllPicksOnTrace(traceInt);
      if (pickState) {
        upsertLocalPick(traceInt, pickState.time);
        queueUpsert(traceInt, pickState.time);
        return;
      }
      queueDelete(traceInt);
    }

    function recordManualPickHistory(changes) {
      if (applyingManualPickHistory) return;
      const normalizedChanges = Array.isArray(changes)
        ? changes
          .map((change) => {
            const trace = change?.trace | 0;
            const before = cloneManualPick(change?.before);
            const after = cloneManualPick(change?.after);
            if (!Number.isFinite(trace) || manualPickStateEquals(before, after)) {
              return null;
            }
            return { trace, before, after };
          })
          .filter(Boolean)
        : [];
      if (normalizedChanges.length === 0) {
        updateManualPickHistoryButtons();
        return;
      }
      pushManualPickHistoryEntry({
        sectionKey: currentManualPickHistorySectionKey(),
        changes: normalizedChanges,
      });
    }

    function applyManualPickHistoryEntry(entry, direction) {
      if (!entry || !Array.isArray(entry.changes) || entry.changes.length === 0) return false;
      const sectionKey = currentManualPickHistorySectionKey();
      if (entry.sectionKey !== sectionKey) {
        console.warn('[PICKS] manual pick history scope mismatch; clearing history', {
          entrySectionKey: entry.sectionKey,
          currentSectionKey: sectionKey,
        });
        clearManualPickHistory();
        return false;
      }
      const useBefore = direction === 'undo';
      applyingManualPickHistory = true;
      try {
        for (const change of entry.changes) {
          const trace = change.trace | 0;
          const targetState = useBefore ? change.before : change.after;
          applyManualPickState(trace, targetState);
        }
        schedulePickOverlayUpdate();
        return true;
      } finally {
        applyingManualPickHistory = false;
      }
    }

    function undoManualPickEdit() {
      if (manualPickUndoStack.length === 0) {
        updateManualPickHistoryButtons();
        return;
      }
      const entry = manualPickUndoStack.pop();
      if (applyManualPickHistoryEntry(entry, 'undo')) {
        manualPickRedoStack.push(entry);
      }
      updateManualPickHistoryButtons();
    }

    function redoManualPickEdit() {
      if (manualPickRedoStack.length === 0) {
        updateManualPickHistoryButtons();
        return;
      }
      const entry = manualPickRedoStack.pop();
      if (applyManualPickHistoryEntry(entry, 'redo')) {
        manualPickUndoStack.push(entry);
      }
      updateManualPickHistoryButtons();
    }

    function getSelectedFbModelId() {
      const sel = document.getElementById('fbpick_model_select');
      if (!sel || !sel.value) return null;
      return sel.value;
    }

    function updateFbModelStorageFromSelect() {
      const selected = getSelectedFbModelId();
      if (!selected) {
        localStorage.removeItem(FBPICK_MODEL_STORAGE_KEY);
        return;
      }
      localStorage.setItem(FBPICK_MODEL_STORAGE_KEY, selected);
    }

    function onFbModelChange() {
      updateFbModelStorageFromSelect();
      recomputeFbPicks();
    }

    async function initFbModelSelect() {
      const sel = document.getElementById('fbpick_model_select');
      const label = document.getElementById('fbpick_model_select_label');
      if (!sel || !label) {
        console.warn('[FBPICK] model select UI is missing');
        return;
      }

      sel.innerHTML = '';
      sel.disabled = true;
      label.style.display = 'none';

      try {
        const response = await fetch('/fbpick_models');
        if (!response.ok) {
          throw new Error(`/fbpick_models failed (${response.status})`);
        }

        const payload = await response.json();
        const models = Array.isArray(payload?.models) ? payload.models : [];
        const defaultModelId = typeof payload?.default_model_id === 'string'
          ? payload.default_model_id
          : null;

        if (models.length === 0) {
          console.warn('[FBPICK] model list is empty');
          return;
        }

        for (const model of models) {
          const modelId = typeof model?.id === 'string' ? model.id : '';
          if (!modelId) continue;
          const usesOffset = model?.uses_offset === true;
          const optionText = usesOffset ? `${modelId} (offset)` : modelId;
          sel.appendChild(new Option(optionText, modelId));
        }

        if (sel.options.length === 0) {
          console.warn('[FBPICK] model list had no valid ids');
          return;
        }

        const savedModelId = localStorage.getItem(FBPICK_MODEL_STORAGE_KEY);
        const hasSaved = savedModelId && Array.from(sel.options).some((opt) => opt.value === savedModelId);
        const hasDefault = defaultModelId && Array.from(sel.options).some((opt) => opt.value === defaultModelId);

        if (hasSaved) {
          sel.value = savedModelId;
        } else if (hasDefault) {
          sel.value = defaultModelId;
        } else {
          sel.selectedIndex = 0;
        }

        updateFbModelStorageFromSelect();
        sel.disabled = false;
        label.style.display = '';
      } catch (err) {
        console.warn('[FBPICK] failed to initialize model list', err);
      }
    }

    let dragOverride = null; // 一時上書き（'pan' を入れる）

    function getBaseDragMode() {
      return (dragBase === 'pan') ? 'pan' : 'zoom';
    }

    function setBaseDragMode(mode) {
      const next = (mode === 'pan') ? 'pan' : 'zoom';
      if (dragBase !== next) {
        dragBase = next;
        try { localStorage.setItem(cfg.LS_KEYS.DRAG_BASE, next); } catch (_) { }
        applyDragMode();
      }
    }

    function getOverrideDragMode() {
      if (dragOverride) return dragOverride;
      if (isPickMode) return false;
      return null;
    }

    function effectiveDragMode() {
      const over = getOverrideDragMode();
      if (over !== null && over !== undefined) return over;
      return getBaseDragMode();
    }
    function applyDragMode() {
      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;

      // 既存関数：'zoom' or 'pan' を返す（ピック中は false を返していた）
      const dm = effectiveDragMode();

      //  - ピック中（dm===false）のときは dragmode は 'zoom' のままにして
      //    x/y の fixedrange を true にしてドラッグ無効化
      //  - Alt パンや通常時は fixedrange を false に戻す
      if (dm === false) {
        // ピック中（ドラッグ禁止）
        safeRelayout(plotDiv, {
          dragmode: 'zoom',                  // 値は何でもOKだが合法値を入れておく
          'xaxis.fixedrange': true,
          'yaxis.fixedrange': true,
        });
        applyDragMode._last = 'pick-locked';
        return;
      }

      // 通常 or Alt パン（ドラッグ可能）
      if (applyDragMode._last === dm) return; // 余計な relayout を避ける
      applyDragMode._last = dm;
      safeRelayout(plotDiv, {
        dragmode: dm,                        // 'zoom' or 'pan'
        'xaxis.fixedrange': false,
        'yaxis.fixedrange': false,
      });
    }

    function safeRelayout(gd, props) {
      suppressRelayout = true;
      try {
        const result = Plotly.relayout(gd, props);
        if (result && typeof result.finally === 'function') {
          return result.finally(() => {
            suppressRelayout = false;
          });
        }
        suppressRelayout = false;
        return result;
      } catch (err) {
        suppressRelayout = false;
        throw err;
      }
    }

    function schedulePickOverlayUpdate() {
      pickOverlayDirty = true;
      if (pickOverlayRaf !== 0) return;
      pickOverlayRaf = requestAnimationFrame(() => {
        pickOverlayRaf = 0;
        flushPickOverlayUpdate();
        if (pickOverlayDirty && !isRelayouting) {
          schedulePickOverlayUpdate();
        }
      });
    }

    function resolvePendingPickYRange(plotDiv) {
      const yaRange = plotDiv?._fullLayout?.yaxis?.range;
      if (
        Array.isArray(yaRange) &&
        yaRange.length === 2 &&
        Number.isFinite(yaRange[0]) &&
        Number.isFinite(yaRange[1])
      ) {
        return {
          yMin: Math.min(yaRange[0], yaRange[1]),
          yMax: Math.max(yaRange[0], yaRange[1]),
        };
      }

      const dt = Number(window.defaultDt ?? defaultDt) || 1;
      const wy0 = Number(latestWindowRender?.y0);
      const wy1 = Number(latestWindowRender?.y1);
      if (Number.isFinite(wy0) && Number.isFinite(wy1)) {
        return {
          yMin: Math.min(wy0 * dt, wy1 * dt),
          yMax: Math.max(wy0 * dt, wy1 * dt),
        };
      }

      return { yMin: 0, yMax: 0 };
    }

    function flushPickOverlayUpdate() {
      if (!pickOverlayDirty) return;
      if (isRelayouting) return;

      const plotDiv = document.getElementById('plot');
      if (!plotDiv) {
        pickOverlayDirty = false;
        return;
      }
      let xMin = null;
      let xMax = null;

      // Prefer the rendered window span so panning does not trim pick traces to viewport only.
      if (Number.isFinite(renderedStart) && Number.isFinite(renderedEnd)) {
        xMin = Math.floor(Math.min(renderedStart, renderedEnd));
        xMax = Math.ceil(Math.max(renderedStart, renderedEnd));
      }

      if (!(Number.isFinite(xMin) && Number.isFinite(xMax) && xMin <= xMax)) {
        const wx0 = Number(latestWindowRender?.x0);
        const wx1 = Number(latestWindowRender?.x1);
        if (Number.isFinite(wx0) && Number.isFinite(wx1)) {
          xMin = Math.floor(Math.min(wx0, wx1));
          xMax = Math.ceil(Math.max(wx0, wx1));
        }
      }

      if (!(Number.isFinite(xMin) && Number.isFinite(xMax) && xMin <= xMax)) {
        const xaRange = plotDiv?._fullLayout?.xaxis?.range;
        if (
          Array.isArray(xaRange) &&
          xaRange.length === 2 &&
          Number.isFinite(xaRange[0]) &&
          Number.isFinite(xaRange[1])
        ) {
          xMin = Math.floor(Math.min(xaRange[0], xaRange[1]));
          xMax = Math.ceil(Math.max(xaRange[0], xaRange[1]));
        }
      }

      if (!(Number.isFinite(xMin) && Number.isFinite(xMax) && xMin <= xMax)) {
        pickOverlayDirty = false;
        return;
      }

      const showPred = !!document.getElementById('showFbPred')?.checked;
      const [manualPickTr, predPickTr] = buildPickMarkerTraces({
        manualPicks: picks,
        predicted: showPred ? predictedPicks : [],
        xMin,
        xMax,
        showPredicted: showPred,
        timeTransform: window.pickRawTimeToDisplayTime,
      });
      const { yMin, yMax } = resolvePendingPickYRange(plotDiv);
      const pendingPickTr = buildPendingPickMarkerTrace({
        pending: getPendingPickOverlayState(),
        yMin,
        yMax,
      });
      pickOverlayDirty = false;
      const resolver = (typeof resolvePickTraceIndices === 'function')
        ? resolvePickTraceIndices
        : window.resolvePickTraceIndices;
      if (typeof resolver !== 'function') {
        console.warn('[PICKS] resolvePickTraceIndices is not available');
        return;
      }
      const { manualIdx, predIdx, pendingIdx } = resolver(plotDiv);
      if (manualIdx < 0 || predIdx < 0 || pendingIdx < 0) {
        console.warn('[PICKS] pick traces are missing; overlay restyle skipped');
        if (typeof renderLatestView === 'function') renderLatestView();
        return;
      }
      const restyleResult = Plotly.restyle(plotDiv, {
        x: [manualPickTr.x, predPickTr.x, pendingPickTr.x],
        y: [manualPickTr.y, predPickTr.y, pendingPickTr.y],
        visible: [true, !!showPred, !!pendingPickTr.visible],
        mode: [manualPickTr.mode, predPickTr.mode, pendingPickTr.mode],
        marker: [manualPickTr.marker, predPickTr.marker, pendingPickTr.marker],
        line: [manualPickTr.line, predPickTr.line, pendingPickTr.line],
      }, [manualIdx, predIdx, pendingIdx]);
      if (restyleResult && typeof restyleResult.catch === 'function') {
        restyleResult.catch((err) => console.warn('pick overlay restyle failed', err));
      }
    }

    // 統一キー関数（FB予測キャッシュ用）
    function fbCacheKey(fileId, k1, layer, pKey, modelId) {
      return `${fileId}|${k1}|${layer}|${pKey ?? 'raw'}|${modelId ?? ''}`;
    }

    // 一時上書きの適用
    function setAltPan(on) {
      const next = on ? 'pan' : null;
      if (dragOverride !== next) {
        dragOverride = next;
        applyDragMode();
      }
    }

    function currentDesiredMode() {
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        return typeof compareCurrentDesiredMode === 'function' ? compareCurrentDesiredMode() : null;
      }
      const win = currentVisibleWindow();
      const plotDiv = document.getElementById('plot');
      if (!win || !plotDiv) return null;
      const wantWig = wantWiggleForWindow({
        tracesVisible: win.nTraces,
        samplesVisible: win.nSamples,
        widthPx: plotDiv.clientWidth || 1,
      });
      return wantWig ? 'wiggle' : 'heatmap';
    }

    function isResetRelayout(ev) {
      if (!ev || typeof ev !== 'object') return false;

      if (ev['xaxis.autorange'] === true || ev['yaxis.autorange'] === true) {
        return true;
      }

      for (const key of Object.keys(ev)) {
        if (!/^(xaxis|yaxis)(\d+)?\.autorange$/.test(key)) continue;
        if (ev[key] === true) return true;
      }

      const keys = Object.keys(ev);
      const onlyRangeLikeKeys = keys.length > 0 && keys.every((key) =>
        /^(xaxis|yaxis)(\d+)?\.(range(\[\d+\])?|autorange)$/.test(key)
      );
      if (!onlyRangeLikeKeys) return false;

      const xRange = readAxisRange(ev, 'xaxis');
      const yRange = readAxisRange(ev, 'yaxis');
      if (!xRange || !yRange || !Array.isArray(sectionShape) || sectionShape.length < 2) {
        return false;
      }

      const totalTraces = Number(sectionShape[0]);
      const totalSamples = Number(sectionShape[1]);
      const dt = window.defaultDt ?? defaultDt;
      if (!Number.isFinite(totalTraces) || !Number.isFinite(totalSamples) || totalTraces <= 0 || totalSamples <= 0) {
        return false;
      }
      if (!Number.isFinite(dt) || dt <= 0) return false;

      const xMin = Math.min(xRange[0], xRange[1]);
      const xMax = Math.max(xRange[0], xRange[1]);
      const yMin = Math.min(yRange[0], yRange[1]);
      const yMax = Math.max(yRange[0], yRange[1]);
      const fullX = xMin <= 0 && xMax >= (totalTraces - 1);
      const fullY = yMin <= 0 && yMax >= ((totalSamples - 1) * dt);
      if (fullX && fullY) return true;

      return false;
    }

    function requestWindowFetch({ immediate = false } = {}) {
      const plotDiv = document.getElementById('plot');
      if (!plotDiv) return;

      if (!immediate) {
        scheduleWindowFetch();
        return;
      }

      const now = Date.now();
      if (now - lastImmediateWindowFetchAt < RESET_FETCH_DEDUP_MS) return;
      lastImmediateWindowFetchAt = now;

      if (typeof scheduleWindowFetch?.cancel === 'function') {
        scheduleWindowFetch.cancel();
      }
      if (typeof scheduleWindowFetch?.flush === 'function') {
        scheduleWindowFetch.flush();
        return;
      }

      const result = fetchWindowAndPlot();
      if (result && typeof result.catch === 'function') {
        result.catch((err) => console.warn('Window fetch failed', err));
      }
    }

    function flushPendingResetFetchIfNeeded() {
      if (!pendingResetFetch) return;
      if (suppressRelayout || isRelayouting) return;
      pendingResetFetch = false;
      checkModeFlipAndRefetch({ immediate: true });
    }

    function checkModeFlipAndRefetch({ immediate = false } = {}) {
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        if (typeof checkCompareModeFlipAndRefetch === 'function') {
          checkCompareModeFlipAndRefetch({ immediate });
        }
        return;
      }
      const desired = currentDesiredMode();
      if (!desired) return;
      const cur = (latestWindowRender && latestWindowRender.mode) || null;
      const plotDiv = document.getElementById('plot');
      const win = currentVisibleWindow();
      if (!plotDiv || !win) return;

      if (desired === 'wiggle') {
        // wiggle は step=1 前提。モード不一致 or step不一致 or ウィンドウ外なら再フェッチ
        const needFresh =
          !latestWindowRender ||
          cur !== 'wiggle' ||
          latestWindowRender.stepX !== 1 ||
          latestWindowRender.stepY !== 1 ||
          latestWindowRender.x0 > win.x0 || latestWindowRender.x1 < win.x1 ||
          latestWindowRender.y0 > win.y0 || latestWindowRender.y1 < win.y1;
        if (needFresh) requestWindowFetch({ immediate });
        return;
      }

      // heatmap の場合：現在の可視窓と描画サイズから必要 step を再計算して比較
      const { step_x, step_y } = computeStepsForWindow({
        tracesVisible: win.nTraces,
        samplesVisible: win.nSamples,
        widthPx: plotDiv.clientWidth || 1,
        heightPx: plotDiv.clientHeight || 1,
      });

      const needFresh =
        !latestWindowRender ||
        cur !== 'heatmap' ||
        latestWindowRender.stepX !== step_x ||
        latestWindowRender.stepY !== step_y ||
        latestWindowRender.x0 > win.x0 || latestWindowRender.x1 < win.x1 ||
        latestWindowRender.y0 > win.y0 || latestWindowRender.y1 < win.y1;

      if (needFresh) requestWindowFetch({ immediate });
    }

    // （任意：すでに入れているならそのままでOK）


    // 入力系にフォーカスがある時は無効
    function canUseGlobalHotkey() {
      const el = document.activeElement;
      const tag = el?.tagName;
      return !(tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable);
    }

    let shortcutsDialogLastActiveElement = null;

    function getShortcutsOverlay() {
      return document.getElementById('viewerShortcutsOverlay');
    }

    function getShortcutsDialog() {
      return document.getElementById('viewerShortcutsDialog');
    }

    function getShortcutsButton() {
      return document.getElementById('viewerShortcutsButton');
    }

    function isShortcutsDialogOpen() {
      const overlay = getShortcutsOverlay();
      return !!overlay && overlay.hidden === false;
    }

    function isShortcutHelpOpenKey(event) {
      const keyRaw = typeof event?.key === 'string' ? event.key : '';
      return keyRaw === '?' || (keyRaw === '/' && event.shiftKey);
    }

    function openShortcutsDialog() {
      const overlay = getShortcutsOverlay();
      const dialog = getShortcutsDialog();
      const closeBtn = document.getElementById('viewerShortcutsClose');
      if (!overlay || !dialog) return;
      if (isShortcutsDialogOpen()) return;
      shortcutsDialogLastActiveElement = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
      overlay.hidden = false;
      window.requestAnimationFrame(() => {
        if (closeBtn && typeof closeBtn.focus === 'function') {
          closeBtn.focus({ preventScroll: true });
          return;
        }
        if (typeof dialog.focus === 'function') {
          dialog.focus({ preventScroll: true });
        }
      });
    }

    function closeShortcutsDialog(options) {
      const overlay = getShortcutsOverlay();
      if (!overlay || overlay.hidden) return;
      overlay.hidden = true;
      const restoreFocus = !options || options.restoreFocus !== false;
      if (!restoreFocus) return;
      const fallback = getShortcutsButton();
      const target =
        shortcutsDialogLastActiveElement &&
        shortcutsDialogLastActiveElement.isConnected &&
        typeof shortcutsDialogLastActiveElement.focus === 'function'
          ? shortcutsDialogLastActiveElement
          : fallback;
      shortcutsDialogLastActiveElement = null;
      if (target && typeof target.focus === 'function') {
        target.focus({ preventScroll: true });
      }
    }

    const HOTKEY_PAN_MIN_TRACES = 10;
    const HOTKEY_PAN_RATIO = 0.10;
    const HOTKEY_GAIN_TARGET = 1.2;
    const HOTKEY_GAIN_QUANTILE = 0.99;
    const HOTKEY_GAIN_MAX_SAMPLES = 50000;

    function getSectionFullRanges() {
      if (!Array.isArray(sectionShape) || sectionShape.length < 2) return null;
      const totalTraces = Number(sectionShape[0]);
      const totalSamples = Number(sectionShape[1]);
      const dt = window.defaultDt ?? defaultDt;
      if (!Number.isFinite(totalTraces) || totalTraces <= 0) return null;
      if (!Number.isFinite(totalSamples) || totalSamples <= 0) return null;
      if (!Number.isFinite(dt) || dt <= 0) return null;
      return {
        xRange: [0, totalTraces - 1],
        yRange: [(totalSamples - 1) * dt, 0],
      };
    }

    function getCurrentXAxisRangeForHotkey() {
      const plotDiv = document.getElementById('plot');
      const xRange = plotDiv?._fullLayout?.xaxis?.range;
      if (
        Array.isArray(xRange) &&
        xRange.length === 2 &&
        Number.isFinite(xRange[0]) &&
        Number.isFinite(xRange[1])
      ) {
        return [Number(xRange[0]), Number(xRange[1])];
      }
      if (
        Array.isArray(savedXRange) &&
        savedXRange.length === 2 &&
        Number.isFinite(savedXRange[0]) &&
        Number.isFinite(savedXRange[1])
      ) {
        return [Number(savedXRange[0]), Number(savedXRange[1])];
      }
      if (Number.isFinite(renderedStart) && Number.isFinite(renderedEnd)) {
        return [Number(renderedStart), Number(renderedEnd)];
      }
      const full = getSectionFullRanges();
      return full ? [full.xRange[0], full.xRange[1]] : null;
    }

    function shiftAndClampXRange(range, delta) {
      if (!Array.isArray(range) || range.length !== 2) return null;
      if (!Number.isFinite(range[0]) || !Number.isFinite(range[1])) return null;

      const reversed = range[0] > range[1];
      let lo = Math.min(range[0], range[1]) + delta;
      let hi = Math.max(range[0], range[1]) + delta;

      const totalTraces = Number(sectionShape?.[0]);
      if (Number.isFinite(totalTraces) && totalTraces > 0) {
        const minTrace = 0;
        const maxTrace = totalTraces - 1;
        const span = hi - lo;

        if (span >= (maxTrace - minTrace)) {
          lo = minTrace;
          hi = maxTrace;
        } else {
          if (lo < minTrace) {
            hi += (minTrace - lo);
            lo = minTrace;
          }
          if (hi > maxTrace) {
            lo -= (hi - maxTrace);
            hi = maxTrace;
          }
          lo = Math.max(minTrace, lo);
          hi = Math.min(maxTrace, hi);
        }
      }

      return reversed ? [hi, lo] : [lo, hi];
    }

    function queueHotkeyXAxisPan(delta) {
      if (!Number.isFinite(delta) || delta === 0) return;
      pendingHotkeyXPanDelta += delta;
      if (hotkeyXPanRaf !== 0) return;
      hotkeyXPanRaf = requestAnimationFrame(() => {
        hotkeyXPanRaf = 0;
        const totalDelta = pendingHotkeyXPanDelta;
        pendingHotkeyXPanDelta = 0;
        if (!totalDelta) return;

        const plotDiv = document.getElementById('plot');
        const curRange = getCurrentXAxisRangeForHotkey();
        if (!plotDiv || !curRange || typeof window.Plotly?.relayout !== 'function') return;
        const nextRange = shiftAndClampXRange(curRange, totalDelta);
        if (!nextRange) return;

        const relayoutResult = window.Plotly.relayout(plotDiv, { 'xaxis.range': nextRange });
        if (relayoutResult && typeof relayoutResult.catch === 'function') {
          relayoutResult.catch((err) => console.warn('Hotkey pan relayout failed', err));
        }
      });
    }

    function runHotkeyFullView() {
      const full = getSectionFullRanges();
      if (!full) {
        console.warn('[HOTKEY] Full view skipped: section metadata is not ready');
        return;
      }

      savedXRange = [full.xRange[0], full.xRange[1]];
      savedYRange = [Math.max(full.yRange[0], full.yRange[1]), Math.min(full.yRange[0], full.yRange[1])];
      forceFullExtentOnce = true;

      const plotDiv = document.getElementById('plot');
      if (plotDiv && typeof window.Plotly?.relayout === 'function') {
        const relayoutResult = window.Plotly.relayout(plotDiv, {
          'xaxis.range': full.xRange,
          'yaxis.range': full.yRange,
        });
        if (relayoutResult && typeof relayoutResult.catch === 'function') {
          relayoutResult.catch((err) => console.warn('Hotkey full-view relayout failed', err));
        }
      }

      requestWindowFetch({ immediate: true });
    }

    function changeSectionByDelta(delta) {
      if (!Number.isFinite(delta) || delta === 0) return;
      const idx = getCurrentKey1Index();
      if (idx < 0) return;
      const changed = selectKey1Index(idx + delta);
      if (!changed) return;
      onKey1Change({ immediate: true }).catch((err) => console.warn('Section delta change failed', err));
    }

    function runHotkeyStepKey1(delta) {
      changeSectionByDelta(delta);
    }

    function toggleLayerByHotkey() {
      const sel = document.getElementById('layerSelect');
      if (!sel) return;
      if (sel.options.length > 1) {
        sel.value = sel.value === 'raw' ? sel.options[1].value : 'raw';
        drawSelectedLayer();
      }
    }

    function setGainFromHotkey(nextGain) {
      const gainEl = document.getElementById('gain');
      if (!gainEl) return false;
      let value = Number(nextGain);
      if (!Number.isFinite(value)) return false;

      const min = parseFloat(gainEl.min);
      const max = parseFloat(gainEl.max);
      if (Number.isFinite(min)) value = Math.max(min, value);
      if (Number.isFinite(max)) value = Math.min(max, value);

      gainEl.value = String(value);
      onGainChange();
      return true;
    }

    function estimateAbsQuantileFromWindow(win, quantile = HOTKEY_GAIN_QUANTILE, maxSamples = HOTKEY_GAIN_MAX_SAMPLES) {
      if (!win || typeof win !== 'object') return null;

      const valuesI8 = win.valuesI8 instanceof Int8Array ? win.valuesI8 : null;
      const valuesRaw = !valuesI8 && win.values && win.values.length != null ? win.values : null;
      const source = valuesI8 || valuesRaw;
      if (!source || source.length === 0) return null;

      const stride = Math.max(1, Math.ceil(source.length / Math.max(1, maxSamples)));
      const scale = valuesI8 ? (Number(win.scale) || 1) : 1;
      if (valuesI8 && (!Number.isFinite(scale) || scale === 0)) return null;

      const sampled = [];
      for (let i = 0; i < source.length; i += stride) {
        const raw = Number(source[i]);
        if (!Number.isFinite(raw)) continue;
        const amp = Math.abs(valuesI8 ? (raw / scale) : raw);
        if (!Number.isFinite(amp)) continue;
        sampled.push(amp);
      }
      if (sampled.length === 0) return null;

      sampled.sort((a, b) => a - b);
      const qRaw = Number(quantile);
      const q = Number.isFinite(qRaw) ? Math.max(0, Math.min(1, qRaw)) : HOTKEY_GAIN_QUANTILE;
      const idx = Math.min(sampled.length - 1, Math.floor((sampled.length - 1) * q));
      return sampled[idx];
    }

    function runHotkeyAutoGain() {
      if (!latestWindowRender) {
        console.warn('[HOTKEY] Auto gain skipped: latest window is not available');
        return;
      }
      const q = estimateAbsQuantileFromWindow(latestWindowRender);
      if (!Number.isFinite(q) || q <= 0) {
        console.warn('[HOTKEY] Auto gain skipped: amplitude estimate is invalid');
        return;
      }
      const nextGain = HOTKEY_GAIN_TARGET / q;
      if (!setGainFromHotkey(nextGain)) {
        console.warn('[HOTKEY] Auto gain skipped: gain slider is not available');
      }
    }

    function handleViewerHotkey(e) {
      const keyRaw = typeof e.key === 'string' ? e.key : '';
      const key = keyRaw.toLowerCase();

      if (isShortcutsDialogOpen()) {
        if (keyRaw === 'Escape') {
          e.preventDefault();
          closeShortcutsDialog();
        } else if (isShortcutHelpOpenKey(e)) {
          e.preventDefault();
        }
        return;
      }

      if (!canUseGlobalHotkey()) return;

      if (isShortcutHelpOpenKey(e)) {
        e.preventDefault();
        if (e.repeat) return;
        openShortcutsDialog();
        return;
      }

      if (e.ctrlKey || e.metaKey) {
        const isUndo = key === 'z' && !e.shiftKey;
        const isRedo = (key === 'z' && e.shiftKey) || key === 'y';
        if (isUndo || isRedo) {
          e.preventDefault();
          if (isUndo) {
            undoManualPickEdit();
          } else {
            redoManualPickEdit();
          }
        }
        return;
      }
      if (!plotHover) return;

      const allowWithAlt = key === 'g' && e.shiftKey;
      if (e.altKey && !allowWithAlt) return;

      if (keyRaw === 'ArrowLeft' || keyRaw === 'ArrowRight') {
        e.preventDefault();
        const xRange = getCurrentXAxisRangeForHotkey();
        if (!xRange) return;
        const visibleWidth = Math.max(1, Math.abs(xRange[1] - xRange[0]) + 1);
        const step = Math.max(HOTKEY_PAN_MIN_TRACES, Math.round(visibleWidth * HOTKEY_PAN_RATIO));
        queueHotkeyXAxisPan(keyRaw === 'ArrowLeft' ? -step : step);
        return;
      }

      if (key === 'r') {
        e.preventDefault();
        if (e.repeat) return;
        runHotkeyFullView();
        return;
      }

      if (key === 'a' || key === 'd' || keyRaw === 'PageUp' || keyRaw === 'PageDown') {
        e.preventDefault();
        if (e.repeat) return;
        const delta = (key === 'a' || keyRaw === 'PageUp') ? -1 : 1;
        runHotkeyStepKey1(delta);
        return;
      }

      if (key === 'g') {
        e.preventDefault();
        if (e.repeat) return;
        if (e.shiftKey) {
          if (!setGainFromHotkey(1)) {
            console.warn('[HOTKEY] Gain reset skipped: gain slider is not available');
          }
        } else {
          runHotkeyAutoGain();
        }
        return;
      }

      if (key === 'p') {
        e.preventDefault();
        if (e.repeat) return;
        togglePickMode();
        return;
      }

      if (key === 'n') {
        e.preventDefault();
        if (e.repeat) return;
        toggleLayerByHotkey();
      }
    }

    // Alt 押してる間だけ pan
    window.addEventListener('keydown', (e) => {
      if (isShortcutsDialogOpen() || !canUseGlobalHotkey()) return;
      if (e.key === 'Alt' || e.altKey) setAltPan(true);
    });
    window.addEventListener('keydown', handleViewerHotkey);
    window.addEventListener('keyup', (e) => {
      if (e.key === 'Alt' || !e.altKey) setAltPan(false);
    });

    // 取りこぼし対策
    window.addEventListener('blur', () => setAltPan(false));
    document.addEventListener('visibilitychange', () => { if (document.hidden) setAltPan(false); });
    window.addEventListener('pointerup', (e) => { if (!e.altKey) setAltPan(false); });

    function applyServerDt(obj) {
      const dtSec =
        obj && typeof obj.dt === 'number' && isFinite(obj.dt) && obj.dt > 0
          ? obj.dt
          : null;

      if (dtSec === null) return;

      const prev =
        typeof window.defaultDt === 'number' && isFinite(window.defaultDt)
          ? window.defaultDt
          : defaultDt;

      // dt が実際に変わった時だけ更新＆Yレンジ初期化
      if (!Number.isFinite(prev) || Math.abs(prev - dtSec) > 1e-12) {
        defaultDt = dtSec;
        window.defaultDt = dtSec;
        cfg.setDefaultDt(dtSec);
        savedYRange = null;
      }
    }

    const COLORMAPS = {
      Greys: 'Greys',
      RdBu: 'RdBu',
      BWR: [[0, 'blue'], [0.5, 'white'], [1, 'red']],
      Cividis: 'Cividis',
      Jet: 'Jet',
    };
    window.COLORMAPS = COLORMAPS;

    (function () {
      const saved = localStorage.getItem('gain');
      const el = document.getElementById('gain');
      const disp = document.getElementById('gain_display');
      if (el && disp) {
        const val = saved !== null ? parseFloat(saved) : parseFloat(el.value);
        el.value = val; disp.textContent = `${val}×`;
      }
    })();

    (function () {
      const sel = document.getElementById('colormap');
      const chk = document.getElementById('cmReverse');
      if (sel) {
        const saved = localStorage.getItem('colormap');
        if (saved) sel.value = saved;
      }
      if (chk) {
        const savedRev = localStorage.getItem('cmReverse');
        if (savedRev !== null) chk.checked = savedRev === 'true';
      }
    })();

    (function restoreFbUi() {
      const sig = localStorage.getItem('sigma_ms_max');
      if (sig !== null) {
        const s = document.getElementById('sigma_ms_max');
        if (s) s.value = parseFloat(sig);
      }
      const pm = localStorage.getItem('pick_method');
      if (pm) {
        const sel = document.getElementById('pick_method');
        if (sel) sel.value = pm;
      }
      const sh = localStorage.getItem('showFbPred');
      if (sh !== null) {
        const c = document.getElementById('showFbPred');
        if (c) c.checked = (sh === 'true');
      }
    })();

    (function restoreSnapUi() {
      const m = localStorage.getItem('snap_mode');
      if (m) {
        const el = document.getElementById('snap_mode');
        if (el) el.value = m;
      }
      const w = localStorage.getItem('snap_ms');
      if (w) {
        const el = document.getElementById('snap_ms');
        if (el) el.value = w;
      }
      const r = localStorage.getItem('snap_refine');
      if (r) {
        const el = document.getElementById('snap_refine');
        if (el) el.value = r;
      }
    })();

    // ★★★ FB確率取得：レイヤ/パイプライン対応（既存のジョブAPIを使用）
    function getCachedFbEntry(cacheKey, method, sigmaMsMax) {
      const entry = fbPredCache.get(cacheKey);
      if (!entry) return null;
      if (entry.method !== method) return null;
      if (Math.abs(entry.sigma_ms_max - sigmaMsMax) > 1e-9) return null;
      return entry;
    }

    function installUnifiedClickRouter(plotDiv) {
      if (!plotDiv || plotDiv.__unifiedClickRouter) return;

      const onClick = (e) => {
        if (e.detail > 1) return; // keep Plotly double-click reset
        if (e.button !== 0) return;
        if (!isPickMode) return;
        if (dragOverride === 'pan' || e.altKey) return;
        if (isRelayouting) return;

        const activeEl = document.activeElement;
        if (activeEl && activeEl !== document.body) {
          const tag = activeEl.tagName;
          if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || activeEl.isContentEditable) {
            return;
          }
        }

        const env = getPlotEnv();
        if (!env) return;
        const { rect, m } = env;
        const innerX = e.clientX - rect.left;
        const innerY = e.clientY - rect.top;
        if (innerX < m.l || innerX > m.l + m.w || innerY < m.t || innerY > m.t + m.h) {
          return;
        }

        const tr = traceAtPixel(e.clientX);
        const tSec = timeAtPixel(e.clientY);
        if (!Number.isFinite(tr) || !Number.isFinite(tSec)) return;

        e.stopImmediatePropagation();
        e.stopPropagation();
        e.preventDefault();

        const maybePromise = handlePickNormalized({
          trace: snapTraceFromDataX(tr),
          time: snapTimeFromDataY(tSec),
          shiftKey: !!e.shiftKey,
          ctrlKey: !!e.ctrlKey,
          altKey: !!e.altKey,
        });
        if (maybePromise && typeof maybePromise.catch === 'function') {
          maybePromise.catch(err => console.warn('handlePickNormalized failed', err));
        }
      };

      const remover = () => {
        plotDiv.removeEventListener('click', onClick, true);
        if (plotDiv.__unifiedClickRouter === remover) {
          plotDiv.__unifiedClickRouter = null;
        }
      };
      plotDiv.addEventListener('click', onClick, { capture: true });
      plotDiv.__unifiedClickRouter = remover;
    }

    function attachPickListeners(plotDiv) {
      // relayout ハンドラは一度だけインストール
      installPlotlyViewportHandlersOnce();

      // 既存の pick/hover ハンドラを外す
      plotDiv.removeAllListeners('plotly_click');
      plotDiv.removeAllListeners('plotly_hover');
      plotDiv.removeAllListeners('plotly_unhover');

      if (plotDiv._genericClickHandler) {
        plotDiv.removeEventListener('click', plotDiv._genericClickHandler);
        plotDiv._genericClickHandler = null;
      }
      if (plotDiv._captureShiftHandler) {
        plotDiv.removeEventListener('click', plotDiv._captureShiftHandler, true);
        plotDiv._captureShiftHandler = null;
      }

      // 黒いホバーボックスと同じ値を保持
      plotDiv.on('plotly_hover', (e) => {
        const p = e?.points?.[0];
        if (!p) return;
        lastHover = {
          x: p.x,
          y: p.y,
          meta: Number.isFinite(p.data?.meta) ? p.data.meta : null,
          t: performance.now(),
        };
      });
      plotDiv.on('plotly_unhover', () => { lastHover = null; });

      installUnifiedClickRouter(plotDiv);
    }

    function recomputeFbPicks() {
      const slider = document.getElementById('key1_slider');
      if (!slider) return;
      const idx0 = parseInt(slider.value, 10);
      const keyAtNow = key1Values[idx0];
      const layerNow = (document.getElementById('layerSelect')?.value) || 'raw';
      const pKeyNow = window.latestPipelineKey || null;
      const modelIdNow = getSelectedFbModelId();
      const method = document.getElementById('pick_method').value;
      const sigma = Number(document.getElementById('sigma_ms_max').value) || 20;
      const cacheKeyStr = fbCacheKey(currentFileId, keyAtNow, layerNow, pKeyNow, modelIdNow);
      const cached = getCachedFbEntry(cacheKeyStr, method, sigma);
      if (cached) {
        predictedPicks = (cached.picks || []).slice();
        currentFbKey = keyAtNow;
        currentFbLayer = layerNow;
        currentFbPipelineKey = pKeyNow;
        renderLatestView();
        return;
      }
      predictFromFb();
    }

    function onSigmaChange() {
      localStorage.setItem('sigma_ms_max', document.getElementById('sigma_ms_max').value);
      recomputeFbPicks();
    }
    function onPickMethodChange() {
      localStorage.setItem('pick_method', document.getElementById('pick_method').value);
      recomputeFbPicks();
    }

    async function predictFromFb() {
      const statusToken = beginOpStatus('predict', 'checking cache...');
      const idx0 = parseInt(document.getElementById('key1_slider').value, 10);
      const keyAtStart = key1Values[idx0];
      const layerAtStart = (document.getElementById('layerSelect')?.value) || 'raw';
      const pipelineKeyAtStart = window.latestPipelineKey || null;
      const modelIdAtStart = getSelectedFbModelId();
      const method = document.getElementById('pick_method').value;
      const sigmaMax = Number(document.getElementById('sigma_ms_max').value) || 20;
      const cacheKeyStr = fbCacheKey(currentFileId, keyAtStart, layerAtStart, pipelineKeyAtStart, modelIdAtStart);

      const cached = getCachedFbEntry(cacheKeyStr, method, sigmaMax);
      const reqToken = ++fbPredReqId;
      const btn = document.getElementById('predictFbBtn');
      if (btn) btn.disabled = true;

      try {
        if (!currentFileId) {
          const message = 'file_id not loaded';
          if (setOpStatusIfCurrent('predict', statusToken, 'error', message)) {
            pushToast({
              level: 'error',
              title: 'Predict failed',
              message,
              sticky: true,
              dedupeKey: 'predict:error',
            });
          }
          return;
        }

        if (cached) {
          predictedPicks = (cached.picks || []).slice();
          currentFbKey = keyAtStart;
          currentFbLayer = layerAtStart;
          currentFbPipelineKey = pipelineKeyAtStart;
          if (setOpStatusIfCurrent('predict', statusToken, 'success', `cached (${predictedPicks.length} picks)`)) {
            pushToast({
              level: 'success',
              title: 'Predict complete',
              message: `Cached result (${predictedPicks.length} picks).`,
              dedupeKey: 'predict:success',
            });
          }
          renderLatestView();
          return;
        }

        if (!setOpStatusIfCurrent('predict', statusToken, 'running', 'requesting FB picks...')) return;

        const body = {
          file_id: currentFileId,
          key1: keyAtStart,
          key1_byte: currentKey1Byte,
          key2_byte: currentKey2Byte,
          method,
          sigma_ms_max: sigmaMax,
          model_id: modelIdAtStart,
        };
        if (layerAtStart && layerAtStart !== 'raw' && pipelineKeyAtStart) {
          body.pipeline_key = pipelineKeyAtStart;
          body.tap_label = layerAtStart;
        }

        const res = await fetch('/fbpick_predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const detail = await readErrorDetail(res);
          const message = detail || `fbpick_predict failed (${res.status})`;
          throw new Error(message);
        }

        const data = await res.json();
        const picks = Array.isArray(data?.picks)
          ? data.picks.map(p => ({ trace: p.trace, time: p.time }))
          : [];

        const idxNow = parseInt(document.getElementById('key1_slider').value, 10);
        const keyNow = key1Values[idxNow];
        const layerNow = (document.getElementById('layerSelect')?.value) || 'raw';
        const pipelineKeyNow = window.latestPipelineKey || null;
        const modelIdNow = getSelectedFbModelId();
        if (reqToken !== fbPredReqId ||
          keyNow !== keyAtStart ||
          layerNow !== layerAtStart ||
          pipelineKeyNow !== pipelineKeyAtStart ||
          modelIdNow !== modelIdAtStart) {
          return;
        }

        predictedPicks = picks.slice();
        currentFbKey = keyAtStart;
        currentFbLayer = layerAtStart;
        currentFbPipelineKey = pipelineKeyAtStart;
        fbPredCache.set(cacheKeyStr, {
          picks: picks.slice(),
          method,
          sigma_ms_max: sigmaMax,
        });

        if (setOpStatusIfCurrent('predict', statusToken, 'success', `${picks.length} picks`)) {
          pushToast({
            level: 'success',
            title: 'Predict complete',
            message: `${picks.length} picks`,
            dedupeKey: 'predict:success',
          });
        }
        renderLatestView();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (setOpStatusIfCurrent('predict', statusToken, 'error', message)) {
          pushToast({
            level: 'error',
            title: 'Predict failed',
            message,
            sticky: true,
            dedupeKey: 'predict:error',
          });
        }
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    function onScalingChange() {
      const sel = document.getElementById('scalingMode');
      const rawValue = (sel && typeof sel.value === 'string') ? sel.value.toLowerCase() : 'amax';
      currentScaling = rawValue === 'tracewise' ? 'tracewise' : 'amax';
      if (sel) sel.value = currentScaling;
      try { localStorage.setItem('scaling_mode', currentScaling); } catch (_) {}
      if (windowFetchCtrl) {
        windowFetchCtrl.abort();
        windowFetchCtrl = null;
      }
      scheduleWindowFetch();
    }

    function onWiggleDensityChange() {
      const el = document.getElementById('wiggle_density');
      const v = parseFloat(el?.value);
      if (!Number.isFinite(v) || v <= 0) return;
      const stored = cfg.setWiggleDensity(v);
      if (el) el.value = stored.toFixed(2);
      WIGGLE_DENSITY_THRESHOLD = stored;
      // Apply now: re-evaluate window mode and redraw
      renderLatestView();
      scheduleWindowFetch();
    }

    // Restore UI on load
    (function restoreWiggleUi() {
        const el = document.getElementById('wiggle_density');
        if (!el) return;
        const min = parseFloat(el.min) || 0.02;
        const max = parseFloat(el.max) || 0.30;
        let v;
        if (window.cfg && typeof window.cfg.getWiggleDensity === 'function') {
            v = window.cfg.getWiggleDensity();
          } else {
            const saved = localStorage.getItem('wiggle_density');
            const p = parseFloat(saved);
            v = Number.isFinite(p) ? p : (parseFloat(el.value) || 0.10);
          }
        v = Math.min(max, Math.max(min, v));
        el.value = v.toFixed(2);
        WIGGLE_DENSITY_THRESHOLD = v;
      })();

    (function hookFetchOnce() {
        if (window.__fetchHooked) return;
        window.__fetchHooked = true;
        const orig = window.fetch;
        window.fetch = async function (...args) {
          const url = String(args?.[0] || '');
          const method = (args?.[1]?.method || 'GET').toUpperCase();
          const t0 = performance.now();
          try {
            const res = await orig.apply(this, args);
            if (/\/picks(\?|$)/.test(url)) {
              console.debug(`📡 FETCH ${method} ${url} → ${res.status} (${(performance.now() - t0).toFixed(0)}ms)`);
              console.trace('  ↳ stack @picks'); // 重要：誰が呼んだか
            } else if (/get_section_window_bin/.test(url)) {
              console.debug(`📡 FETCH ${method} ${url} → ${res.status} (${(performance.now() - t0).toFixed(0)}ms)`);
            }
            return res;
          } catch (e) {
            console.debug(`📡 FETCH ${method} ${url} ✖`, e);
            throw e;
          }
        };
      })();


    function onGainChange() {
      const el = document.getElementById('gain');
      const val = el ? el.value : '1';
      document.getElementById('gain_display').textContent = `${parseFloat(val)}×`;
      localStorage.setItem('gain', val);
      scheduleRestyle();
    }

    function onColormapChange() {
      const sel = document.getElementById('colormap');
      const chk = document.getElementById('cmReverse');
      if (sel) localStorage.setItem('colormap', sel.value);
      if (chk) localStorage.setItem('cmReverse', chk.checked);
      scheduleRestyle();
    }

    function togglePickMode() {
      isPickMode = !isPickMode;
      const btn = document.getElementById('pickModeBtn');
      btn.textContent = isPickMode ? 'Pick Mode: ON' : 'Pick Mode: OFF';
      btn.classList.toggle('active', isPickMode);
      clearPendingPickState('toggle-pick-mode');
      const plotDiv = document.getElementById('plot');
      if (plotDiv) {
        const relayoutResult = safeRelayout(plotDiv, {
          clickmode: clickModeForCurrentState(),
        });
        if (relayoutResult && typeof relayoutResult.finally === 'function') {
          relayoutResult.finally(() => {
            applyDragMode();
          });
        } else {
          applyDragMode();
        }
      } else {
        applyDragMode();
      }
      schedulePickOverlayUpdate();
    }

    function getTraceSamplesForProcessingRange(trace, rowLoIncl, rowHiIncl) {
      const win = latestWindowRender;
      if (!win || win.effectiveLayer !== 'raw') return null;

      const rows = Number(win.shape?.[0] ?? 0);
      const cols = Number(win.shape?.[1] ?? 0);
      if (!rows || !cols) return null;

      const x0 = Number(win.x0);
      const stepX = Number(win.stepX) || 1;
      const col = Math.round((trace - x0) / stepX);
      if (col < 0 || col >= cols) return null;

      const loReq = Math.round(Number(rowLoIncl));
      const hiReq = Math.round(Number(rowHiIncl));
      if (!Number.isFinite(loReq) || !Number.isFinite(hiReq)) return null;
      const lo = Math.max(0, Math.min(rows - 1, loReq));
      const hi = Math.max(0, Math.min(rows - 1, hiReq));
      if (hi < lo) return null;

      const valuesI8 = win.valuesI8 instanceof Int8Array ? win.valuesI8 : null;
      const valuesF32 = !valuesI8 && win.values && win.values.length != null ? win.values : null;
      if (!valuesI8 && !valuesF32) return null;

      const total = rows * cols;
      if (valuesI8 && valuesI8.length < total) return null;
      if (valuesF32 && valuesF32.length < total) return null;

      const scale = Number(win.scale) || 1;
      const len = hi - lo + 1;
      const samples = new Float32Array(len);
      for (let r = lo; r <= hi; r++) {
        const idx = r * cols + col;
        samples[r - lo] = valuesI8 ? (valuesI8[idx] / scale) : valuesF32[idx];
      }

      const sampleStep = Number(win.stepY) || 1;

      return {
        samples,
        sampleStart: (Number(win.y0) || 0) + lo * sampleStep,
        sampleStep,
      };
    }

    // 3-point parabolic interpolation around index i (for peak/trough). Returns a float index.
    function parabolicRefine(arr, i) {
      const n = arr.length;
      const ii = Math.max(1, Math.min(n - 2, i | 0));
      const y1 = arr[ii - 1], y2 = arr[ii], y3 = arr[ii + 1];
      const denom = (y1 - 2 * y2 + y3);
      if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return ii; // near-flat → skip
      const delta = 0.5 * (y1 - y3) / denom;          // typically within ~[-0.5, 0.5]
      if (!Number.isFinite(delta) || Math.abs(delta) > 0.6) return ii;   // outlier move → skip
      const xhat = ii + delta;
      if (!Number.isFinite(xhat)) return ii;
      return Math.max(0, Math.min(n - 1, xhat));
    }

    // Upward zero-crossing linear interpolation near i (for rise). Returns a float index.
    function zeroCrossRefine(arr, i) {
      const n = arr.length;
      let i0 = Math.max(0, Math.min(n - 2, i | 0));
      let i1 = i0 + 1;

      // Prefer a nearby upward zero-crossing if present
      if (!(arr[i0] <= 0 && arr[i1] > 0)) {
        if (i0 > 0 && (arr[i0 - 1] <= 0 && arr[i0] > 0)) { i0 = i0 - 1; i1 = i0 + 1; }
        else if (i1 < n - 1 && (arr[i1] <= 0 && arr[i1 + 1] > 0)) { i0 = i1; i1 = i0 + 1; }
        else return i; // no clear upward zero-cross nearby → keep integer index
      }

      const dy = (arr[i1] - arr[i0]);
      if (!Number.isFinite(dy) || Math.abs(dy) < 1e-12) return i; // near-flat → skip
      const frac = (0 - arr[i0]) / dy;                // ∈ [0, 1]
      const xhat = i0 + frac;
      if (!Number.isFinite(xhat)) return i;
      return Math.max(0, Math.min(n - 1, xhat));
    }

    // Adjust a pick's time (seconds) on a given trace to the nearest feature in ±window.
    // Snap is supported only when raw layer data is available in latestWindowRender.
    function adjustPickToFeature(trace, timeSec) {
      const mode = (document.getElementById('snap_mode')?.value) || 'none';
      if (mode === 'none') return timeSec;

      const refineMode = (document.getElementById('snap_refine')?.value) || 'none';

      const dt = (window.defaultDt ?? defaultDt);
      if (!Number.isFinite(dt) || dt <= 0) return timeSec;
      const win = latestWindowRender;
      if (!win || win.effectiveLayer !== 'raw') return timeSec;
      const rows = Number(win.shape?.[0] ?? 0);
      const cols = Number(win.shape?.[1] ?? 0);
      if (!rows || !cols) return timeSec;
      const sampleStartWin = Number(win.y0) || 0;
      const sampleStep = Number(win.stepY) || 1;
      const dtEff = dt * sampleStep;
      if (!Number.isFinite(dtEff) || dtEff <= 0) return timeSec;

      const i0Abs = Math.round(timeSec / dt);
      if (!Number.isFinite(i0Abs)) return timeSec;
      const i0Win = Math.round((i0Abs - sampleStartWin) / sampleStep);
      if (!Number.isFinite(i0Win) || i0Win < 0 || i0Win >= rows) return timeSec;

      // ±window in samples
      const ms = parseFloat(document.getElementById('snap_ms')?.value) || 4;
      const rad = Math.max(1, Math.round((ms / 1000) / dtEff));

      const loWin = Math.max(0, i0Win - rad);
      const hiWin = Math.min(rows - 1, i0Win + rad);
      const sliceLo = Math.max(0, loWin - 1);
      const sliceHi = Math.min(rows - 1, hiWin + 1);
      const traceView = getTraceSamplesForProcessingRange(trace, sliceLo, sliceHi);
      if (!traceView || !traceView.samples) return timeSec;
      const arr = traceView.samples;
      if (arr.length < 3) return timeSec;

      const i0 = i0Win - sliceLo;
      if (!Number.isFinite(i0) || i0 < 0 || i0 >= arr.length) return timeSec;

      // keep one-sample margin for central differences
      const lo = Math.max(1, loWin - sliceLo);
      const hi = Math.min(arr.length - 2, hiWin - sliceLo);
      if (lo > hi) return timeSec;

      let idx = i0;

      if (mode === 'peak') {
        // 近傍の「局所最大」(arr[i-1] <= arr[i] >= arr[i+1]) のうち i0 に最も近い点
        let best = null, bestDist = Infinity;
        for (let i = lo; i <= hi; i++) {
          if (arr[i] >= arr[i - 1] && arr[i] >= arr[i + 1]) {
            const d = Math.abs(i - i0);
            if (d < bestDist) { bestDist = d; best = i; }
          }
        }
        if (best != null) idx = best;
        else {
          // 局所最大がない時のフォールバック：従来どおり最大値へ
          let vmax = -Infinity;
          for (let i = lo; i <= hi; i++) { const v = arr[i]; if (v > vmax) { vmax = v; idx = i; } }
        }
      } else if (mode === 'trough') {
        // 近傍の「局所最小」(arr[i-1] >= arr[i] <= arr[i+1]) のうち最も近い点
        let best = null, bestDist = Infinity;
        for (let i = lo; i <= hi; i++) {
          if (arr[i] <= arr[i - 1] && arr[i] <= arr[i + 1]) {
            const d = Math.abs(i - i0);
            if (d < bestDist) { bestDist = d; best = i; }
          }
        }
        if (best != null) idx = best;
        else {
          // フォールバック：従来どおり最小値へ
          let vmin = Infinity;
          for (let i = lo; i <= hi; i++) { const v = arr[i]; if (v < vmin) { vmin = v; idx = i; } }
        }
      } else if (mode === 'rise') {
        // 「最も近い上りエッジ」を優先：ゼロクロス↑があれば最優先、なければ最大正勾配
        let best = null, bestDist = Infinity;
        for (let i = lo; i < hi; i++) {
          if (arr[i] <= 0 && arr[i + 1] > 0) {
            const cand = (Math.abs(arr[i]) < Math.abs(arr[i + 1])) ? i : i + 1;
            const d = Math.abs(cand - i0);
            if (d < bestDist) { bestDist = d; best = cand; }
          }
        }
        if (best != null) idx = best;
        else {
          let smax = -Infinity;
          for (let i = lo; i <= hi; i++) {
            const s = arr[i + 1] - arr[i - 1];
            if (s > 0 && s > smax) { smax = s; idx = i; }
          }
        }
      }

      let idxFloat = idx;
      if (mode === 'peak' || mode === 'trough') {
        if (refineMode === 'parabolic') idxFloat = parabolicRefine(arr, idx);
      } else if (mode === 'rise') {
        if (refineMode === 'zc') idxFloat = zeroCrossRefine(arr, idx);
      }

      const idxAbs = traceView.sampleStart + idxFloat * traceView.sampleStep;
      return idxAbs * dt;
    }

    function wantWiggleForWindow({ tracesVisible, samplesVisible, widthPx }) {
      const density = tracesVisible / Math.max(1, widthPx);
      if (density >= WIGGLE_DENSITY_THRESHOLD) return false;
      if ((tracesVisible * samplesVisible) > WIGGLE_MAX_POINTS) return false;
      return true;
    }

    function currentVisibleWindow() {
      if (!sectionShape) return null;
      const [totalTraces, totalSamples] = sectionShape;

      // X range
      let x0, x1;
      if (forceFullExtentOnce) {
        x0 = 0; x1 = totalTraces - 1;
      } else if (savedXRange && savedXRange.length === 2) {
        const minX = Math.min(savedXRange[0], savedXRange[1]);
        const maxX = Math.max(savedXRange[0], savedXRange[1]);
        x0 = Math.floor(minX);
        x1 = Math.ceil(maxX);
      } else if (typeof renderedStart === 'number' && typeof renderedEnd === 'number') {
        x0 = renderedStart;
        x1 = renderedEnd;
      } else {
        x0 = 0;
        x1 = totalTraces - 1;
      }

      x0 = Math.max(0, Math.floor(x0));
      x1 = Math.min(totalTraces - 1, Math.ceil(x1));
      if (x1 < x0) [x0, x1] = [x1, x0];

      const spanX = Math.max(1, x1 - x0 + 1);
      const padX = (!forceFullExtentOnce && !!savedXRange)
        ? Math.max(1, Math.floor(spanX * 0.5))
        : 0;
      x0 = Math.max(0, x0 - padX);
      x1 = Math.min(totalTraces - 1, x1 + padX);

      // Y range
      const dtBase = window.defaultDt ?? defaultDt;
      let yMinSec, yMaxSec;
      if (!forceFullExtentOnce && savedYRange && savedYRange.length === 2) {
        yMinSec = Math.min(savedYRange[0], savedYRange[1]);
        yMaxSec = Math.max(savedYRange[0], savedYRange[1]);
      } else {
        yMinSec = 0;
        yMaxSec = (totalSamples - 1) * dtBase;
      }

      let y0 = Math.floor(yMinSec / dtBase);
      let y1 = Math.ceil(yMaxSec / dtBase);
      y0 = Math.max(0, y0);
      y1 = Math.min(totalSamples - 1, y1);
      if (y1 < y0) [y0, y1] = [y1, y0];

      const spanY = Math.max(1, y1 - y0 + 1);
      const padY = (!forceFullExtentOnce && !!savedYRange)
        ? Math.max(1, Math.floor(spanY * 0.1))
        : 0;
      y0 = Math.max(0, y0 - padY);
      y1 = Math.min(totalSamples - 1, y1 + padY);

      // one-shot full-extent is consumed here
      if (forceFullExtentOnce) forceFullExtentOnce = false;

      return {
        x0, x1, y0, y1,
        nTraces: x1 - x0 + 1,
        nSamples: y1 - y0 + 1,
      };
    }

    function updateKey1Display() {
      updateSectionNavigation({ syncDisplay: true });
    }

    function stepKey1(delta) {
      const idx = getCurrentKey1Index();
      if (idx < 0) return false;
      return selectKey1Index(idx + delta);
    }

    function setKey1SliderMax(max) {
      const { slider } = getSectionNavNodes();
      if (!slider) return;
      slider.max = String(Math.max(0, Number(max) || 0));
    }

    async function fetchKey1Values() {
      const res = await fetch(`/get_key1_values?file_id=${currentFileId}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`);
      if (res.ok) {
        const data = await res.json();
        key1Values = data.values;
        setKey1SliderMax(key1Values.length - 1);
        document.getElementById('key1_slider').value = 0;
        clearSectionNavValidation();
        updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
      } else {
        key1Values = [];
        updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
      }
    }

    async function loadSettings() {
      const params = new URLSearchParams(window.location.search);
      currentFileId = params.get('file_id') || localStorage.getItem('file_id') || '';
      currentKey1Byte = parseInt(params.get('key1_byte') || localStorage.getItem('key1_byte') || '189');
      currentKey2Byte = parseInt(params.get('key2_byte') || localStorage.getItem('key2_byte') || '193');
      const scalingParam = (params.get('scaling') || localStorage.getItem('scaling_mode') || 'amax').toLowerCase();
      currentScaling = scalingParam === 'tracewise' ? 'tracewise' : 'amax';
      try { localStorage.setItem('scaling_mode', currentScaling); } catch (_) {}
      const scalingSel = document.getElementById('scalingMode');
      if (scalingSel) scalingSel.value = currentScaling;
      document.getElementById('file_id').value = currentFileId;
      if (!currentFileId) {
        currentFileName = '';
        key1Values = [];
        sectionShape = null;
        updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
        showViewerEmptyState('no-dataset');
        return;
      }
      localStorage.setItem('file_id', currentFileId);
      localStorage.setItem('key1_byte', currentKey1Byte);
      localStorage.setItem('key2_byte', currentKey2Byte);
      await fetchCurrentFileName();
      if (!currentFileName) {
        key1Values = [];
        sectionShape = null;
        updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
        showViewerEmptyState('unavailable');
        return;
      }
      await fetchKey1Values();
      if (!Array.isArray(key1Values) || key1Values.length === 0) {
        sectionShape = null;
        updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
        showViewerEmptyState('unavailable');
        return;
      }
      await fetchSectionMeta();
      if (!Array.isArray(sectionShape) || sectionShape.length < 2) {
        updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
        showViewerEmptyState('unavailable');
        return;
      }
      hideViewerEmptyState();
      await fetchAndPlot();
    }

    async function fetchPicks() {
      if (currentFileName) {
        await reloadPicksForCurrentSection();
        return;
      }
      if (!currentFileId) return;
      const idx = parseInt(document.getElementById('key1_slider').value);
      const key1Val = key1Values?.[idx];
      if (key1Val === undefined) return;
      try {
        const res = await fetch(`/picks?file_id=${currentFileId}&key1=${encodeURIComponent(key1Val)}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`);
        if (res.ok) {
          const data = await res.json();
          picks = (data.picks || []).map(p => ({ trace: p.trace, time: p.time }));
        }
        const [x0, x1] = visibleXRng();
        D('PICKS@fetchPicks', { count: picks.length, vis: [x0, x1],
          visCount: picks.filter(p => Math.round(p.trace) >= x0 && Math.round(p.trace) <= x1).length,
          sample: samplePicks(picks) });
      } catch (e) { console.error('Failed to fetch picks', e); }
    }

    function postPick(trace, time) {
      queueUpsert(trace, time);
      D('PICKS@queuePost', { trace: (trace | 0), time: +time });
    }

    function deletePick(trace) {
      queueDelete(trace);
      D('PICKS@queueDelete', { trace: (trace | 0) });
    }

    // --- filename & picks loader (by filename) ---
    async function fetchSectionMeta() {
      if (!currentFileId) return null;
      try {
        const q = new URLSearchParams({
          file_id: currentFileId,
          key1_byte: String(currentKey1Byte),
          key2_byte: String(currentKey2Byte),
        });
        const res = await fetch(`/get_section_meta?${q.toString()}`);
        if (!res.ok) {
          console.warn('get_section_meta failed', res.status);
          return null;
        }
        const meta = await res.json();
        if (Array.isArray(meta.shape) && meta.shape.length === 2) {
          sectionShape = [Number(meta.shape[0]), Number(meta.shape[1])];
          applyServerDt(meta);
          console.info('[META] sectionShape=', sectionShape, 'dt=', meta.dt);
          return sectionShape;
        }
      } catch (e) {
        console.warn('get_section_meta error', e);
      }
      return null;
    }

    async function fetchCurrentFileName() {
      if (!currentFileId) {
        currentFileName = '';
        return;
      }
      try {
        const r = await fetch(`/file_info?file_id=${encodeURIComponent(currentFileId)}`);
        if (!r.ok) {
          currentFileName = '';
          return;
        }
        const j = await r.json();
        currentFileName = j.file_name || '';
      } catch (e) {
        currentFileName = '';
        console.warn('file_info failed', e);
      }
    }

    async function reloadPicksForCurrentSection(key1IdxOrVal) {
      if (!currentFileId) return [];
      clearManualPickHistory();

      const fileId = currentFileId;
      const idxRaw = Number.isInteger(key1IdxOrVal)
        ? key1IdxOrVal
        : parseInt(document.getElementById('key1_slider').value, 10);
      const idx = Number.isFinite(idxRaw) ? idxRaw : 0;
      const key1Val = key1Values?.[idx];
      if (key1Val === undefined) return [];
      try {
        const r = await fetch(`/picks?file_id=${encodeURIComponent(fileId)}&key1=${encodeURIComponent(key1Val)}&key1_byte=${currentKey1Byte}&key2_byte=${currentKey2Byte}`);
        if (!r.ok) return [];
        const j = await r.json();
        const arr = (j.picks || []).map(p => ({ trace: (p.trace | 0), time: +p.time }));
        picks = arr;
        if (typeof renderLatestView === 'function') {
          try { renderLatestView(); } catch (e) { console.warn('render after picks failed', e); }
        }
        const [x0, x1] = visibleXRng();
        D('PICKS@reload', {
          count: picks.length,
          vis: [x0, x1],
          visCount: picks.filter(p => Math.round(p.trace) >= x0 && Math.round(p.trace) <= x1).length,
          sample: samplePicks(picks)
        });
        return arr;
      } catch (e) {
        console.warn('reload picks failed', e);
        return [];
      }
    }

    async function fetchAndPlot() {
      clearManualPickHistory();
      snapshotAxesRangesFromDOM();
      console.log('--- fetchAndPlot start ---');
      console.time('Total fetchAndPlot');

      if (!currentFileId) {
        showViewerEmptyState('no-dataset');
        return;
      }
      if (!Array.isArray(key1Values) || key1Values.length === 0) {
        showViewerEmptyState('unavailable');
        return;
      }
      hideViewerEmptyState();

      const index = parseInt(document.getElementById('key1_slider').value);
      const key1Val = key1Values[index];

      // ★ FB予測キャッシュ取得：レイヤ＆パイプラインキーでキー統一
      const layerCur = (document.getElementById('layerSelect')?.value) || 'raw';
      const pKeyCur = window.latestPipelineKey || null;
      const modelIdCur = getSelectedFbModelId();
      const methodCur = document.getElementById('pick_method').value;
      const sigmaCur = Number(document.getElementById('sigma_ms_max').value) || 20;
      const cachedEntry = getCachedFbEntry(
        fbCacheKey(currentFileId, key1Val, layerCur, pKeyCur, modelIdCur),
        methodCur,
        sigmaCur,
      );
      predictedPicks = cachedEntry && cachedEntry.picks ? cachedEntry.picks.slice() : [];

      await fetchPicks();

      latestWindowRender = null;
      if (typeof clearCompareRender === 'function') clearCompareRender();
      bumpWindowFetchId();
      if (windowFetchCtrl) {
        windowFetchCtrl.abort();
        windowFetchCtrl = null;
      }
      if (typeof hideLoading === 'function') hideLoading();
      uiResetNonce++;
      if (window.pipelineEvents && typeof window.pipelineEvents.emit === 'function') {
        window.pipelineEvents.emit('section:prepare', {});
      } else if (window.pipelineUI && typeof window.pipelineUI.prepareForNewSection === 'function') {
        window.pipelineUI.prepareForNewSection();
      } else {
        latestTapData = {};
        latestPipelineKey = null;
        const sel = document.getElementById('layerSelect');
        if (sel) {
          sel.innerHTML = '';
          sel.appendChild(new Option('raw', 'raw'));
          sel.value = 'raw';
        }
        if (typeof updateCompareSourceOptions === 'function') updateCompareSourceOptions();
      }

      latestSeismicData = null;
      fetchWindowAndPlot();

      console.timeEnd('Total fetchAndPlot');
      console.log('--- fetchAndPlot end ---');
    }

    function drawSelectedLayer(start = null, end = null) {
      D('DRAW@selectLayer', { layer: document.getElementById('layerSelect')?.value, start, end });
      latestSeismicData = null;
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        if (typeof renderCompareLatestView === 'function') renderCompareLatestView();
        scheduleWindowFetch();
        return;
      }
      renderLatestView();
      scheduleWindowFetch();
    }


    function renderLatestView(startOverride = null, endOverride = null) {
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        if (typeof renderCompareLatestView === 'function') renderCompareLatestView();
        return;
      }
      const sel = document.getElementById('layerSelect');
      const layer = sel ? sel.value : 'raw';
      const slider = document.getElementById('key1_slider');
      const idx = slider ? parseInt(slider.value, 10) : 0;
      const key1Val = key1Values[idx];

      if (latestSeismicData) {
        latestSeismicData = null;
      }

      if (
        latestWindowRender &&
        latestWindowRender.requestedLayer === layer &&
        latestWindowRender.key1 === key1Val
      ) {
        if (layer !== 'raw') {
          const pipelineKeyNow = window.latestPipelineKey || null;
          if ((latestWindowRender.pipelineKey || null) !== (pipelineKeyNow || null)) {
            return;
          }
        }
        if (latestWindowRender.mode === 'wiggle') {
          renderWindowWiggle(latestWindowRender);
        } else {
          renderWindowHeatmap(latestWindowRender);
        }
      }
    }
    function clickModeForCurrentState() {
      return isPickMode ? 'event' : 'event+select';
    }

    function removeAllPicksOnTrace(traceInt) {
      for (let i = picks.length - 1; i >= 0; i--) {
        if ((picks[i].trace | 0) === traceInt) picks.splice(i, 1);
      }
    }

    function upsertLocalPick(traceInt, time) {
      removeAllPicksOnTrace(traceInt);
      picks.push({ trace: traceInt, time: +time });
    }

    function dedupeLocalPicksByTrace() {
      const seen = new Set();
      let removed = 0;
      for (let i = picks.length - 1; i >= 0; i--) {
        const trInt = picks[i].trace | 0;
        if (seen.has(trInt)) {
          picks.splice(i, 1);
          removed += 1;
          continue;
        }
        picks[i].trace = trInt;
        picks[i].time = +picks[i].time;
        seen.add(trInt);
      }
      return removed;
    }

    function toTraceInt(t) {
      // マイナス防止も兼ねる。必要なら上限クリップもここで。
      return Math.max(0, Math.round(t));
    }

    async function displayPickTimeToRawTime(traceInt, displayTime) {
      const converter = window.displayTimeToRawTime;
      if (typeof converter !== 'function') {
        console.warn('LMO pick save skipped because the display/raw time converter is not available.');
        return null;
      }
      let rawTime = converter(traceInt, displayTime);
      if (!Number.isFinite(rawTime) && typeof window.ensureLmoPickOffsetsReady === 'function') {
        await window.ensureLmoPickOffsetsReady();
        rawTime = converter(traceInt, displayTime);
      }
      if (!Number.isFinite(rawTime)) {
        console.warn('LMO pick save skipped because section offsets are not ready.');
        return null;
      }
      return rawTime;
    }

    async function ensureLinePickLmoOffsetsReady() {
      const lmo = typeof window.getCurrentLinearMoveout === 'function'
        ? window.getCurrentLinearMoveout()
        : null;
      if (!lmo || !lmo.enabled) return true;
      if (typeof window.ensureLmoPickOffsetsReady !== 'function') {
        console.warn('LMO line pick skipped because section offsets are not available.');
        return false;
      }
      const ready = await window.ensureLmoPickOffsetsReady();
      if (!ready) {
        console.warn('LMO line pick skipped because section offsets are not ready.');
        return false;
      }
      return true;
    }

    function displayPickTimeToRawTimeSync(traceInt, displayTime) {
      return typeof window.displayTimeToRawTime === 'function'
        ? window.displayTimeToRawTime(traceInt, displayTime)
        : displayTime;
    }

    async function handlePickNormalized({ trace, time, shiftKey, ctrlKey, altKey }) {
      if (isPickMode && dragOverride === 'pan') return;
      if (!Number.isFinite(trace) || !Number.isFinite(time)) return;
      const trInt = toTraceInt(trace);

      if (handlePickNormalized._busy) {
        handlePickNormalized._queued = { trace: trInt, time, shiftKey, ctrlKey, altKey };
        return;
      }
      handlePickNormalized._busy = true;

      try {
        if (!isPickMode) return;

        const deduped = dedupeLocalPicksByTrace();
        if (deduped > 0) {
          console.warn(`[PICKS] removed ${deduped} duplicate local pick(s) before handling input`);
          schedulePickOverlayUpdate();
        }

        console.log('🔥 pick request', { trace, time, shiftKey, ctrlKey, altKey });

        if (ctrlKey) {
          if (deleteRangeStart === null) {
            setDeleteRangeAnchor(trInt, time);
            return;
          }
          const deleteAnchor = normalizeDeleteRangeAnchor(deleteRangeStart);
          clearPendingPickState('delete-range-complete');
          if (!deleteAnchor) return;
          const x0 = deleteAnchor.trace;
          const x1 = trInt;
          const start = Math.min(x0, x1);
          const end = Math.max(x0, x1);
          const toDelete = getLocalPicksInTraceRangeForDisplayDelete(deleteAnchor, trInt, time);
          if (toDelete.length === 0) {
            D('PICKS@handlePickNormalized:delete-range-empty', { range: [start, end] });
            return;
          }
          const deleteTraces = new Set(toDelete.map((p) => p.trace | 0));
          const historyChanges = toDelete.map((p) => {
            const before = cloneManualPick(p);
            return { trace: before.trace, before, after: null };
          });
          const promises = toDelete.map(p => deletePick(p.trace));
          picks = picks.filter(p => !deleteTraces.has(p.trace | 0));
          await Promise.all(promises);
          recordManualPickHistory(historyChanges);
          D('PICKS@handlePickNormalized:line', { range: [start, end], count: picks.length });
          schedulePickOverlayUpdate();
          return;
        }

        if (shiftKey) {
          if (!linePickStart) {
            setLinePickAnchor(trInt, time);
            return;
          }

          const { trace: x0, time: y0 } = linePickStart;
          setLinePickAnchor(trInt, time);
          const x1 = trInt;
          const y1 = time;
          const xStart = Math.round(Math.min(x0, x1));
          const xEnd = Math.round(Math.max(x0, x1));
          const slope = x1 === x0 ? 0 : (y1 - y0) / (x1 - x0);

          if (!(await ensureLinePickLmoOffsetsReady())) return;

          const promises = [];
          const historyChanges = [];
          const linePickWrites = [];
          for (let x = xStart; x <= xEnd; x++) {
            const y = x1 === x0 ? y1 : y0 + slope * (x - x0);
            const snapped = snapTimeFromDataY(y);
            const displayTime = adjustPickToFeature(x, snapped);
            const rawTime = displayPickTimeToRawTimeSync(x, displayTime);
            if (!Number.isFinite(rawTime)) {
              console.warn('LMO line pick skipped because display-to-raw time conversion failed.');
              return;
            }

            const before = getLocalPickOnTraceForDisplayClick(x, displayTime);
            linePickWrites.push({ trace: x, before, rawTime });
          }

          for (const write of linePickWrites) {
            const x = write.trace;
            const before = write.before;
            const hadExisting = !!before;
            removeAllPicksOnTrace(x);
            if (hadExisting) {
              promises.push(deletePick(x));
            }
            upsertLocalPick(x, write.rawTime);
            promises.push(postPick(x, write.rawTime));
            historyChanges.push({
              trace: x,
              before,
              after: { trace: x, time: write.rawTime },
            });
          }
          await Promise.all(promises);
          recordManualPickHistory(historyChanges);
          D('PICKS@handlePickNormalized:line', { range: [xStart, xEnd], count: picks.length });
          schedulePickOverlayUpdate();
          return;
        }

        clearPendingPickState('single-pick');

        const displayTime = adjustPickToFeature(trInt, time);
        const rawTime = await displayPickTimeToRawTime(trInt, displayTime);
        if (rawTime === null) return;

        const promises = [];
        const before = getLocalPickOnTraceForDisplayClick(trInt, displayTime);
        const hadExisting = !!before;
        removeAllPicksOnTrace(trInt);
        if (hadExisting) {
          promises.push(deletePick(trInt));
        }
        upsertLocalPick(trInt, rawTime);
        promises.push(postPick(trInt, rawTime));
        await Promise.all(promises);
        recordManualPickHistory([{ trace: trInt, before, after: { trace: trInt, time: rawTime } }]);
        D('PICKS@handlePickNormalized:single', {
          add: { trace: trInt, time: rawTime },
          displayTime,
          count: picks.length,
        });
        schedulePickOverlayUpdate();
      } finally {
        handlePickNormalized._busy = false;
        const next = handlePickNormalized._queued;
        handlePickNormalized._queued = null;
        if (next) setTimeout(() => handlePickNormalized(next), 0);
      }
    }

    async function handleRelayout(ev) {
      if (suppressRelayout) return;
      if (!ev || typeof ev !== 'object') return;
      if (typeof isCompareModeEnabled === 'function' && isCompareModeEnabled()) {
        if (typeof handleCompareRelayout === 'function') {
          await handleCompareRelayout(ev);
        }
        return;
      }

      flushPendingResetFetchIfNeeded();

      D('RELAYOUT@begin', { keys: Object.keys(ev), isRelayouting, pickMode: isPickMode });

      const gd = document.getElementById('plot');
      if (!gd) return;

      // range 更新
      const xRange = readAxisRange(ev, 'xaxis');
      if (xRange) {
        savedXRange = xRange;
      }
      const yRange = readAxisRange(ev, 'yaxis');
      if (yRange) {
        const y0 = yRange[0];
        const y1 = yRange[1];
        savedYRange = y0 > y1 ? [y0, y1] : [y1, y0];
      }

      // reset/autorange は debounce をバイパスして即 fetch（shape同期はしない）
      if (isResetRelayout(ev)) {
        await new Promise(r => requestAnimationFrame(r));
        snapshotAxesRangesFromDOM();
        if (isRelayouting) {
          pendingResetFetch = true;
          return;
        }
        pendingResetFetch = false;
        checkModeFlipAndRefetch({ immediate: true });
        return;
      }
      D('RELAYOUT@skip(shape-sync-removed)', viewerState());
    }


    window.addEventListener('DOMContentLoaded', () => {
      const createProgressOverlay = window.createPipelineProgressOverlay;
      if (typeof createProgressOverlay !== 'function') return;
      const progressApi = createProgressOverlay({
        overlayId: 'ppOverlay',
        statusId: 'ppStatus',
        barId: 'ppBarInner',
        cancelId: 'ppCancelBtn',
        onCancel: () => {
          if (window.pipelineUI && typeof window.pipelineUI.cancel === 'function') {
            window.pipelineUI.cancel();
          }
        },
      });
      if (!progressApi) return;
      window.pipelineProgress = progressApi;
    });

    window.addEventListener('DOMContentLoaded', () => {
      console.info('[viewer] DOMContentLoaded hook');
      const fileIdEl = document.getElementById('file_id');
      const slider = document.getElementById('key1_slider');
      const fbpickModelSelect = document.getElementById('fbpick_model_select');
      const plotDiv = document.getElementById('plot');
      const shortcutsButton = getShortcutsButton();
      const shortcutsOverlay = getShortcutsOverlay();
      const shortcutsClose = document.getElementById('viewerShortcutsClose');
      resetOpStatuses();
      updateManualPickHistoryButtons();
      updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });

      if (shortcutsButton) {
        shortcutsButton.addEventListener('click', () => openShortcutsDialog());
      }
      if (shortcutsClose) {
        shortcutsClose.addEventListener('click', () => closeShortcutsDialog());
      }
      if (shortcutsOverlay) {
        shortcutsOverlay.addEventListener('click', (event) => {
          if (event.target === shortcutsOverlay) {
            closeShortcutsDialog();
          }
        });
      }

      if (plotDiv) {
        plotHover = plotDiv.matches(':hover');
        plotDiv.addEventListener('mouseenter', () => { plotHover = true; });
        plotDiv.addEventListener('mouseleave', () => { plotHover = false; });
      }

      const boot = async () => {
        if (fileIdEl && fileIdEl.value && !currentFileId) {
          currentFileId = fileIdEl.value;
        }
        if (isViewerEmptyStateVisible()) {
          return;
        }
        if (!currentFileId) {
          currentFileName = '';
          return;
        }
        if (!currentFileName) {
          await fetchCurrentFileName();
        }
      };

      loadSettings().catch((err) => {
        console.warn('loadSettings failed', err);
        showViewerEmptyState(currentFileId ? 'unavailable' : 'no-dataset');
      }).finally(() => {
        boot().catch((err) => console.warn('initial picks load failed', err));
      });

      initFbModelSelect().catch((err) => console.warn('initFbModelSelect failed', err));

      if (fbpickModelSelect) {
        fbpickModelSelect.addEventListener('change', onFbModelChange);
      }

      if (fileIdEl) {
        fileIdEl.addEventListener('change', async () => {
          const previousFileId = currentFileId;
          currentFileId = fileIdEl.value || '';
          if (currentFileId !== previousFileId && typeof clearLinearMoveoutRuntimeCaches === 'function') {
            clearLinearMoveoutRuntimeCaches();
          }
          key1Values = [];
          sectionShape = null;
          savedXRange = null;
          savedYRange = null;
          renderedStart = null;
          renderedEnd = null;
          latestWindowRender = null;
          if (typeof clearCompareRender === 'function') clearCompareRender();
          bumpWindowFetchId();
          if (typeof hideLoading === 'function') hideLoading();
          latestSeismicData = null;
          latestTapData = {};
          fbPredReqId += 1;

          picks = [];
          predictedPicks = [];
          currentFbKey = null;
          currentFbLayer = 'raw';
          currentFbPipelineKey = null;
          fbPredCache.clear();
          latestPipelineKey = null;
          window.latestPipelineKey = null;
          resetOpStatuses();
          if (windowFetchCtrl) {
            windowFetchCtrl.abort();
            windowFetchCtrl = null;
          }
          updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
          if (!currentFileId) {
            currentFileName = '';
            showViewerEmptyState('no-dataset');
            return;
          }
          await fetchCurrentFileName();
          if (!currentFileName) {
            updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
            showViewerEmptyState('unavailable');
            return;
          }
          await fetchKey1Values();
          if (!Array.isArray(key1Values) || key1Values.length === 0) {
            updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
            showViewerEmptyState('unavailable');
            return;
          }
          await fetchSectionMeta();
          if (!Array.isArray(sectionShape) || sectionShape.length < 2) {
            updateSectionNavigation({ syncDisplay: true, syncJumpInput: true });
            showViewerEmptyState('unavailable');
            return;
          }
          hideViewerEmptyState();
          (typeof fetchAndPlotDebounced?.flush === 'function')
            ? fetchAndPlotDebounced.flush()
            : fetchAndPlot();
        });
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.key === 'Shift') {
        clearPendingPickState('shift-keyup', { keepDeleteRange: true });
      }
    });

    window.addEventListener('lmo:change', () => {
      if (linePickStart || deleteRangeStart !== null) {
        clearPendingPickState('lmo-change');
      }
      schedulePickOverlayUpdate();
    });
