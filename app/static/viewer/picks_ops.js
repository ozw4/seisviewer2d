    function ensureFlushPickOpsDebounced() {
      if (!flushPickOpsDebounced) {
        const debFn = (typeof debounce === 'function') ? debounce : makeDebounced;
        flushPickOpsDebounced = debFn(flushPickOps, 120);
      }
      return flushPickOpsDebounced;
    }

    function _sliderKey1Idx() {
        const el = document.getElementById('key1_slider');
        const v = Number.parseInt(el?.value ?? '0', 10);
        return Number.isFinite(v) ? v : 0;
      }
        function _opKey(fileId, key1Val, key1Byte, trace) {
          return `${fileId}|${key1Val}|${key1Byte}|${trace | 0}`;
        }

    function queueUpsert(trace, time) {
      const tr = (trace | 0);
      const fileId = currentFileId;
      const key1Idx = _sliderKey1Idx();
      const key1Byte = currentKey1Byte;
      const key1Val = key1Values?.[key1Idx];
      if (key1Val === undefined) {
        console.warn('queueUpsert skipped: key1 value unavailable', { key1Idx });
        return;
      }
      const k = _opKey(fileId, key1Val, key1Byte, tr);
      __pickOps.set(k, { op: 'upsert', fileId, key1Val, key1Byte, trace: tr, time: +time });
      const debounced = ensureFlushPickOpsDebounced();
      debounced();
    }

    function queueDelete(trace) {
      const tr = (trace | 0);
      const fileId = currentFileId;
      const key1Idx = _sliderKey1Idx();
      const key1Byte = currentKey1Byte;
      const key1Val = key1Values?.[key1Idx];
      if (key1Val === undefined) {
        console.warn('queueDelete skipped: key1 value unavailable', { key1Idx });
        return;
      }
      const k = _opKey(fileId, key1Val, key1Byte, tr);
      __pickOps.set(k, { op: 'delete', fileId, key1Val, key1Byte, trace: tr });
      const debounced = ensureFlushPickOpsDebounced();
      debounced();
    }
    window.debugDump = () => ({
      viewer: viewerState('DUMP'),
      picks: samplePicks(picks, 20),
      predicted: samplePicks(predictedPicks, 10)
  });
    function logCurrentPicks(opts = {}) {
      const { includePredicted = false, onlyVisible = false } = opts;

      const dt = (window.defaultDt ?? defaultDt);
      const k1Idx = parseInt(document.getElementById('key1_slider').value, 10) | 0;
      const key1Val = key1Values?.[k1Idx];
      const layer = document.getElementById('layerSelect')?.value || 'raw';

      const start = (typeof renderedStart === 'number') ? renderedStart : 0;
      const end = (typeof renderedEnd === 'number') ? renderedEnd : (sectionShape ? sectionShape[0] - 1 : 0);

      const src = includePredicted ? [...picks, ...predictedPicks] : picks;

      const rows = src
        .filter(p => Number.isFinite(p.trace) && Number.isFinite(p.time))
        .map(p => ({
          trace: Math.round(p.trace),
          time_s: +p.time,
          sample_idx: Math.round(p.time / dt),
        }))
        .filter(r => !onlyVisible || (r.trace >= start && r.trace <= end))
        .sort((a, b) => a.trace - b.trace);

      console.log(`[Picks] key1=${key1Val} layer=${layer} count=${rows.length} visibleOnly=${!!onlyVisible}`);
      console.table(rows);
      return rows; // 必要なら呼び出し元で使えるように返す
    }

    // グローバルから呼べるように
    window.logCurrentPicks = logCurrentPicks;
