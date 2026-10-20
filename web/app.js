const api = {
  state: () => fetch('/api/state').then(r => r.json()),
  sources: () => fetch('/api/sources').then(r => r.json()),
  control: (body) => fetch('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(r => r.json()),
};

const el = (id) => document.getElementById(id);
const state = {
  last: null, seen: {}, paused: false, lastFrameAt: 0, lastFrame: 0,
  paintedAt: 0, feedMode: 'stream', feedTries: 0, feedStartedAt: 0, snapTimer: 0,
};

function setStatus(kind, text) {
  const box = el('status');
  box.className = 'status ' + kind;
  box.querySelector('span').textContent = text;
}

/** 画面到底画出来没有: naturalWidth 要首帧解码后才有值, 比各浏览器差异很大的 load 事件可靠。 */
function painted() {
  return el('feed').naturalWidth > 0;
}

function stopSnap() {
  if (state.snapTimer) { clearInterval(state.snapTimer); state.snapTimer = 0; }
}

/** 连续画面流: 首选, 最跟手, 但 Safari 有时会整条流不吐帧。 */
function useStream() {
  stopSnap();
  state.feedMode = 'stream';
  state.feedStartedAt = Date.now();
  state.paintedAt = 0;
  el('feed').src = '/stream?r=' + Date.now();
}

/** 快照轮询: 每 120ms 换一张 /snapshot, 任何浏览器都能画, 稍逊流畅度。 */
function useSnap() {
  if (state.feedMode === 'snap') return;
  stopSnap();
  state.feedMode = 'snap';
  state.feedStartedAt = Date.now();
  state.paintedAt = 0;
  const img = el('feed');
  const tick = () => { img.src = '/snapshot?r=' + Date.now(); };
  tick();
  state.snapTimer = setInterval(tick, 120);
}

/** 一行分段耗时: 卡不卡、卡在哪一段, 在页面上直接能看出来。 */
function stageText(stages) {
  if (!stages || !stages['合计']) return '耗时统计中';
  const overlay = ['缩小', '叠加', '编码'].reduce((a, k) => a + (stages[k] ?? 0), 0);
  return `每帧 ${Math.round(stages['合计'])}ms（推理 ${Math.round(stages['推理'] ?? 0)}ms · 画面 ${Math.round(overlay)}ms）`;
}

function renderMetrics(s) {
  el('m-people').textContent = s.people ?? 0;
  el('m-counting').textContent = s.participants ?? 0;
  el('m-total').textContent = s.total ?? 0;
  const confident = s.total_confident ?? s.total ?? 0;
  const tips = [(s.total_confident ?? s.total) < (s.total ?? 0)
    ? `其中确凿 ${confident} 个, 其余按低置信计入` : '全部为确凿计数'];
  if (s.total_recovered) tips.push(`${s.total_recovered} 个来自能量包络回捞(幅度不足但多通道与节拍都有佐证)`);
  el('m-total').title = tips.join('; ');
  el('m-rope').textContent = s.rope_total ?? 0;
  el('m-fps').textContent = (s.proc_fps ?? 0).toFixed(1);
  el('clock').textContent = `t=${(s.time ?? 0).toFixed(1)}s · 帧 ${s.frame ?? 0} · 画面帧率 ${(s.measured_fps ?? 0).toFixed(1)} · ${stageText(s.stage_ms)}`;
  el('model-name').textContent = s.detector || '';
  if (s.error) setStatus('err', s.error);
  else if (s.paused) setStatus('warn', '已暂停');
  else if (s.running) setStatus('ok', '实时计数中');
  else setStatus('warn', '画面未就绪');
}

function sparkline(canvas, values, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const pts = (values || []).slice(-90).map(Number);
  if (pts.length < 3) return;
  const lo = Math.min(...pts), hi = Math.max(...pts), span = Math.max(1e-6, hi - lo);
  ctx.beginPath();
  pts.forEach((v, i) => {
    const x = (w * i) / (pts.length - 1);
    const y = h - 3 - (h - 8) * ((v - lo) / span);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = color + '22'; ctx.fill();
}

function renderCards(s) {
  const tracks = (s.tracks || []).slice().sort((a, b) => b.count - a.count);
  const wrap = el('cards');
  el('cards-empty').hidden = tracks.length > 0;
  wrap.innerHTML = '';
  tracks.forEach((t) => {
    const card = document.createElement('div');
    const muted = !t.participant;
    const provisional = t.verdict === 'provisional';
    const warm = !muted && (provisional || (t.snr ?? 99) < 3 || (t.visibility ?? 1) < 0.75);
    card.className = 'card' + (muted ? ' muted' : warm ? ' warm' : '');
    const color = muted ? '#7f8c9b' : warm ? '#ffce5a' : '#5adc78';
    const badge = muted ? '不计数' : provisional ? '低置信' : warm ? '信号偏弱' : '计数中';
    const stitched = (t.fragments ?? 1) > 1 ? `<span>已接回 ${t.fragments} 段轨迹</span>` : '';
    card.innerHTML = `
      <div class="who"><b>#${t.id}</b><span class="badge">${badge}</span></div>
      <div><div class="count">${t.count}</div><div class="unit">个</div></div>
      <div class="facts">
        <span>节奏 ${t.rate_hz ? t.rate_hz.toFixed(2) + ' Hz' : '—'}</span>
        <span>信噪比 ${t.snr ?? '—'}</span>
        <span>可见率 ${t.visibility != null ? Math.round(t.visibility * 100) + '%' : '—'}</span>
        <span>信号 ${t.signal_source || '—'}</span>
        ${t.filled ? `<span>+${t.filled} 按节奏补</span>` : ''}
        ${t.recovered ? `<span>+${t.recovered} 包络回捞</span>` : ''}
        <span>最近 ${t.last_jump != null ? t.last_jump.toFixed(1) + 's' : '—'}</span>
        ${stitched}
      </div>
      ${t.reason ? `<div class="why">${t.reason}</div>` : ''}
      <canvas></canvas>`;
    wrap.appendChild(card);
    sparkline(card.querySelector('canvas'), t.signal, color);
  });
}

function renderNotes(s) {
  const list = el('notes');
  const notes = s.warnings || [];
  el('notes-empty').hidden = notes.length > 0;
  list.innerHTML = notes.map((n) => `<li class="${n.level}">${n.text}</li>`).join('');
}

function renderEvents(s) {
  const list = el('events');
  const events = (s.events || []).slice(-14).reverse();
  list.innerHTML = events.map((e) => {
    const kind = e.kind === 'bump' ? '双摇过绳' : e.kind === 'gap' ? `空档 ${e.gap}×周期` : '起跳';
    const tag = e.recovered ? '·包络回捞' : '';
    return `<li><b>#${e.track}</b><span>${kind}${tag}</span><span>${(e.time ?? 0).toFixed(2)}s</span><span>幅度 ${e.amplitude ?? '—'}</span></li>`;
  }).join('');
}

async function poll() {
  try {
    const s = await api.state();
    state.last = s;
    if (painted()) state.paintedAt = Date.now();
    renderMetrics(s);
    renderCards(s);
    renderNotes(s);
    renderEvents(s);
    if (s.frame !== state.lastFrame) { state.lastFrame = s.frame; state.lastFrameAt = Date.now(); }
    const stalled = Date.now() - state.lastFrameAt > 6000;
    // 后台在出帧而页面一个像素都没画出来: 先把流重连两次, 还不行就退化成快照轮询
    if (s.running && !state.paintedAt && Date.now() - state.feedStartedAt > 5000 && state.feedMode === 'stream') {
      if (state.feedTries < 2) { state.feedTries += 1; useStream(); }
      else useSnap();
    }
    let feedMsg = '';
    if (s.error) feedMsg = s.error;
    else if (stalled) feedMsg = '画面已经停止更新';
    else if (!state.paintedAt && Date.now() - state.feedStartedAt > 5000) feedMsg = '画面没能显示，点下面按钮重试';
    el('feed-error').hidden = !feedMsg;
    if (feedMsg) el('feed-error-text').textContent = feedMsg;
    if (state.feedMode === 'snap') el('clock').textContent += ' · 画面=快照模式（连续流已自动重连失败）';
  } catch (err) {
    setStatus('err', '与后台断开');
  }
}

function debounce(fn, ms) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function loadSources() {
  const { sources, current } = await api.sources();
  const sel = el('source');
  sel.innerHTML = sources.map((s) => {
    const label = s.kind === 'camera' ? s.label : `${s.label} (${s.size_mb} MB)`;
    return `<option value="${s.id}">${label}</option>`;
  }).join('');
  const hit = sources.find((s) => String(s.id) === String(current));
  if (hit) sel.value = hit.id;
}

function wire() {
  el('feed').onload = () => { state.paintedAt = Date.now(); };
  el('btn-pause').onclick = async () => {
    const next = state.last?.paused ? 'resume' : 'pause';
    await api.control({ action: next });
    el('btn-pause').textContent = next === 'pause' ? '继续' : '暂停';
  };
  el('btn-reset').onclick = () => api.control({ action: 'reset' });
  el('btn-retry').onclick = () => { state.feedTries = 0; useStream(); };
  // 切源会让旧的来流作废: 主动重连, 否则画面停在上一源的最后一帧上不动
  el('source').onchange = (e) => {
    api.control({ action: 'source', source: e.target.value });
    state.feedTries = 0;
    useStream();
  };
  el('policy').onchange = (e) => api.control({ action: 'config', policy: e.target.value });
  el('imgsz').onchange = (e) => api.control({ action: 'config', imgsz: Number(e.target.value) });
  el('strictness').onchange = (e) => api.control({ action: 'config', min_visibility: Number(e.target.value) });
  const pushSens = debounce((v) => api.control({ action: 'config', sensitivity: Number(v) }), 250);
  el('sensitivity').oninput = (e) => { el('sens-val').textContent = e.target.value; pushSens(e.target.value); };
  const pushConf = debounce((v) => api.control({ action: 'config', conf: Number(v) }), 300);
  el('conf').oninput = (e) => { el('conf-val').textContent = Number(e.target.value).toFixed(2); pushConf(e.target.value); };
}

(async function start() {
  wire();
  state.lastFrameAt = Date.now();
  state.feedStartedAt = Date.now();
  await loadSources();
  await poll();
  setInterval(poll, 500);
})();
