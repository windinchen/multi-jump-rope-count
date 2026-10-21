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
/** 素材名里的 < & " 等字符必须转义: 直接拼进 <option> 会把下拉框结构整个弄坏, 之后连源都选不了。 */
const esc = (text) => String(text).replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const state = {
  last: null, seen: {}, paused: false, lastFrameAt: 0, lastFrame: 0,
  paintedAt: 0, feedMode: 'stream', feedTries: 0, feedStartedAt: 0, snapTimer: 0,
  controlError: null,
  sourcesAt: 0, sourcesFailed: false, sourceIds: null, sourcePending: null, sourcePendingAt: 0,
};

/** 素材目录多久重扫一次: 素材往文件夹里一丢就能看到, 不用刷新页面。 */
const SOURCES_REFRESH_MS = 5000;
const SOURCES_RETRY_MS = 2000;
/**
 * 切源请求最多"押住"下拉框多久: 押住期间轮询不把选择刷回去, 否则选项会来回闪。
 * 必须比后台的摄像头重试上限(10s)长, 不然一次正常的慢切换会被当成失败弹回上一个素材;
 * 但也不能无限押着: 请求因为网络原因根本没落地时, 下拉框会一直停在一个没生效的素材上。
 */
const SOURCE_PENDING_MS = 12000;

/** 一次起跳的四种来源, 页面上必须分开展示: 实测峰 / 双摇多出的过绳 / 按节奏补 / 包络回捞。 */
const KIND_LABEL = {
  jump: '起跳', bump: '双摇过绳', gap: '空档', interpolated: '按节奏补',
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
  if (s.total_excluded) tips.push(`另有 ${s.total_excluded} 个数在暂停/被剔除的人身上, 暂未计入总计: 停下来超过一个信号窗口会先不算他, 继续跳就接回来`);
  el('m-total').title = tips.join('; ');
  el('m-rope').textContent = s.rope_total ?? 0;
  el('m-fps').textContent = (s.proc_fps ?? 0).toFixed(1);
  el('clock').textContent = `t=${(s.time ?? 0).toFixed(1)}s · 帧 ${s.frame ?? 0} · 画面帧率 ${(s.measured_fps ?? 0).toFixed(1)} · ${stageText(s.stage_ms)}`;
  el('model-name').textContent = s.detector || '';
  // 暂停状态以服务端为准: 按钮文案不能自己翻, 否则控制请求失败时页面和后台就长期对不上
  state.paused = !!s.paused;
  const pauseBtn = el('btn-pause');
  if (pauseBtn) pauseBtn.textContent = state.paused ? '继续' : '暂停';
  if (s.error) setStatus('err', s.error);
  else if (state.controlError && Date.now() - state.controlError.at < 4000) setStatus('err', state.controlError.text);
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

/** 卡片主数字: 与顶部总次数同一口径(实测 + 按节奏补), 否则卡片加起来对不上总数。 */
const totalOf = (t) => (t.count || 0) + (t.filled || 0);

function cardFacts(t) {
  const stitched = (t.fragments ?? 1) > 1 ? `<span>已接回 ${t.fragments} 段轨迹</span>` : '';
  return [
    `节奏 ${t.rate_hz ? t.rate_hz.toFixed(2) + ' Hz' : '—'}`,
    `信噪比 ${t.snr ?? '—'}`,
    `可见率 ${t.visibility != null ? Math.round(t.visibility * 100) + '%' : '—'}`,
    `信号 ${t.signal_source || '—'}`,
    t.filled ? `+${t.filled} 按节奏补` : '',
    t.recovered ? `+${t.recovered} 包络回捞` : '',
    `最近 ${t.last_jump != null ? t.last_jump.toFixed(1) + 's' : '—'}`,
    stitched,
  ].filter(Boolean).join('</span><span>');
}

function buildCard() {
  const card = document.createElement('div');
  card.innerHTML = `
    <div class="who"><b></b><span class="badge"></span></div>
    <div><div class="count"></div><div class="unit">个</div></div>
    <div class="facts"></div>
    <div class="why" hidden></div>
    <canvas></canvas>`;
  card.refs = {
    id: card.querySelector('.who b'),
    badge: card.querySelector('.badge'),
    count: card.querySelector('.count'),
    facts: card.querySelector('.facts'),
    why: card.querySelector('.why'),
    canvas: card.querySelector('canvas'),
  };
  return card;
}

/**
 * 卡片复用而不是每次轮询重建: 原先 500ms 一次的 innerHTML='' 会把 N 个 canvas
 * 连同整棵子树拆掉重建, 页面一直在重排, 光标选中数字也一直被吞掉。
 * 现在按站位号复用节点, 只改变化的文本, 排序靠 insertBefore 调整顺序。
 */
function renderCards(s) {
  const tracks = (s.tracks || []).slice()
    .sort((a, b) => totalOf(b) - totalOf(a) || a.id - b.id);
  const wrap = el('cards');
  el('cards-empty').hidden = tracks.length > 0;
  const map = state.cardMap || (state.cardMap = new Map());
  const live = new Set(tracks.map((t) => t.id));
  for (const [id, node] of [...map]) {
    if (!live.has(id)) { node.remove(); map.delete(id); }
  }
  tracks.forEach((t, index) => {
    let card = map.get(t.id);
    if (!card) {
      card = buildCard();
      map.set(t.id, card);
      wrap.appendChild(card);
    }
    if (wrap.children[index] !== card) wrap.insertBefore(card, wrap.children[index] || null);
    const muted = !t.participant;
    const provisional = t.verdict === 'provisional';
    const warm = !muted && (provisional || (t.snr ?? 99) < 3 || (t.visibility ?? 1) < 0.75);
    card.className = 'card' + (muted ? ' muted' : warm ? ' warm' : '');
    const color = muted ? '#7f8c9b' : warm ? '#ffce5a' : '#5adc78';
    const badge = muted ? '不计数' : provisional ? '低置信' : warm ? '信号偏弱' : '计数中';
    const r = card.refs;
    r.id.textContent = `#${t.id}`;
    r.badge.textContent = badge;
    r.count.textContent = totalOf(t);
    const parts = [`实测 ${t.count || 0}`];
    if (t.carried) parts.push(`含上一次离场前的 ${t.carried}`);
    if (t.filled) parts.push(`按节奏补 ${t.filled}`);
    if (t.recovered) parts.push(`其中包络回捞 ${t.recovered}`);
    if (t.rope != null && t.rope !== totalOf(t)) parts.push(`过绳 ${t.rope}`);
    r.count.title = parts.join(' · ');
    const facts = `<span>${cardFacts(t)}</span>`;
    if (r.facts.innerHTML !== facts) r.facts.innerHTML = facts;
    r.why.hidden = !t.reason;
    if (t.reason && r.why.textContent !== t.reason) r.why.textContent = t.reason;
    sparkline(r.canvas, t.signal, color);
  });
}

function renderNotes(s) {
  const list = el('notes');
  // 换画面源失败要留在这儿说清楚: 后台已经退回上一个源, 只靠一行状态条会被下一轮轮询立刻刷掉
  const notes = [...(s.switch_error ? [{ level: 'err', text: s.switch_error }] : []), ...(s.warnings || [])];
  el('notes-empty').hidden = notes.length > 0;
  const html = notes.map((n) => `<li class="${n.level}">${n.text}</li>`).join('');
  if (list.innerHTML !== html) list.innerHTML = html;
}

function renderEvents(s) {
  const list = el('events');
  const events = (s.events || []).slice(-14).reverse();
  // 以前 interpolated 会落到 else 分支显示成"起跳", 补出来的数看着像真抓到的
  const html = events.map((e) => {
    const kind = e.kind === 'gap' ? `空档 ${e.gap}×周期` : (KIND_LABEL[e.kind] || e.kind || '起跳');
    const tag = e.recovered ? '·包络回捞' : '';
    return `<li><b>#${e.track}</b><span>${kind}${tag}</span><span>${(e.time ?? 0).toFixed(2)}s</span><span>幅度 ${e.amplitude ?? '—'}</span></li>`;
  }).join('');
  if (list.innerHTML !== html) list.innerHTML = html;
}

async function poll() {
  try {
    const s = await api.state();
    state.last = s;
    if (painted()) state.paintedAt = Date.now();
    renderMetrics(s);
    syncControls(s);
    renderCards(s);
    renderNotes(s);
    renderEvents(s);
    maybeLoadSources();
    if (s.frame !== state.lastFrame) { state.lastFrame = s.frame; state.lastFrameAt = Date.now(); }
    // 暂停时帧号本来就不动: 以前 6 秒后会在画面上盖一层"画面已经停止更新"的红条, 像是故障
    const stalled = !s.paused && Date.now() - state.lastFrameAt > 6000;
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
    el('feed-error-text').textContent = feedMsg;
    if (state.feedMode === 'snap') el('clock').textContent += ' · 画面=快照模式（连续流已自动重连失败）';
  } catch (err) {
    setStatus('err', `与后台断开：${err && err.message ? err.message : '请求失败'}`);
  }
}

function debounce(fn, ms) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/**
 * 素材目录列表拉取失败不能把整个页面带走: 以前 loadSources 抛出后 start() 直接中断,
 * 页面白屏且不会重试(素材目录里有半截文件被删时就会撞上)。
 * 失败提示写着"稍后自动重试", 所以这里必须真的重试: 只调用一次的话那句话就是假的。
 */
async function loadSources() {
  state.sourcesAt = Date.now();
  try {
    const body = await api.sources();
    state.sourcesFailed = false;   // 拉通了就按正常节奏重试: 放在下面两个 return 之后会一直停在 2s 快扫
    const sources = (body && body.sources) || [];
    const sel = el('source');
    if (document.activeElement === sel) return;  // 用户正在选源, 别把他手上的选择改掉
    const ids = sources.map((s) => String(s.id)).join('|');
    if (ids === state.sourceIds) return;  // 目录没变就不动 DOM, 免得周期刷新把用户的下拉选择刷掉
    state.sourceIds = ids;
    sel.innerHTML = sources.map((s) => {
      const label = s.kind === 'camera' ? s.label : `${s.label} (${s.size_mb} MB)`;
      return `<option value="${esc(s.id)}">${esc(label)}</option>`;
    }).join('');
    const hit = sources.find((s) => String(s.id) === String(body.current));
    if (hit) sel.value = hit.id;
  } catch (err) {
    state.sourcesFailed = true;
    setStatus('warn', '素材列表读取失败，稍后自动重试');
  }
}

/** 按节奏重扫素材目录: 失败时缩短到 2s 再试, 成功时 5s 一次。 */
function maybeLoadSources() {
  const wait = state.sourcesFailed ? SOURCES_RETRY_MS : SOURCES_REFRESH_MS;
  if (Date.now() - state.sourcesAt < wait) return;
  loadSources();
}

/**
 * 下拉框/滑杆以服务端为准: 换过参数后刷新页面, 控件以前总是退回 HTML 里的默认值,
 * 看着像参数被重置了(实际服务端还留着上次的值)。拖动中的控件不覆盖。
 */
function syncControls(s) {
  const set = (id, value) => {
    if (value == null) return;
    const node = el(id);
    if (!node || document.activeElement === node) return;
    if (node.value !== String(value)) node.value = String(value);
  };
  set('policy', s.policy);
  set('imgsz', s.imgsz);
  set('conf', s.conf);
  set('strictness', s.min_track_visibility);
  set('sensitivity', s.sensitivity);
  if (document.activeElement !== el('sensitivity')) el('sens-val').textContent = Math.round(s.sensitivity ?? 50);
  if (document.activeElement !== el('conf')) el('conf-val').textContent = Number(s.conf ?? 0.35).toFixed(2);
  // 画面源下拉框同样以服务端为准: 切源失败时后台已经退回上一个源, 选择得跟着画面回去,
  // 不然显示的是一个根本没生效的素材。刚点下去还没切完的那次不覆盖, 否则会来回闪。
  const sel = el('source');
  if (sel && document.activeElement !== sel) {
    if (state.sourcePending && Date.now() - state.sourcePendingAt > SOURCE_PENDING_MS) state.sourcePending = null;
    if (state.sourcePending && (s.source === state.sourcePending || s.switch_error)) state.sourcePending = null;
    if (!state.sourcePending && s.source != null) {
      const hit = [...sel.options].find((o) => o.value === String(s.source));
      if (hit && sel.value !== hit.value) sel.value = hit.value;
    }
  }
}

function wire() {
  el('feed').onload = () => { state.paintedAt = Date.now(); };
  // 按钮文案一律由 renderMetrics 按服务端状态刷: 控制请求失败时不会出现"页面显示已暂停、后台还在跑"
  // fetch 在 HTTP 400 上照样 resolve: 服务端拒绝指令(指令拼错/参数越界)要靠返回体里的 error 判断,
  // 只 catch 网络异常的话页面会一直显示"设置已生效"。报错留 4 秒, 免得被下一轮轮询的状态刷掉。
  const control = (body) => api.control(body).then((res) => {
    if (res && res.error) {
      state.controlError = { text: `设置没有生效：${res.error}`, at: Date.now() };
      setStatus('err', state.controlError.text);
    } else if (state.controlError) {
      state.controlError = null;
    }
    return res;
  }).catch((err) => {
    const text = `设置没有生效：${err && err.message ? err.message : '请求失败'}`;
    state.controlError = { text, at: Date.now() };
    setStatus('err', text);
    return null;
  });
  el('btn-pause').onclick = () => control({ action: state.last?.paused ? 'resume' : 'pause' });
  el('btn-reset').onclick = () => control({ action: 'reset' });
  el('btn-retry').onclick = () => { state.feedTries = 0; useStream(); };
  // 切源会让旧的来流作废: 主动重连, 否则画面停在上一源的最后一帧上不动
  el('source').onchange = (e) => {
    state.sourcePending = e.target.value;   // 这次切换落地之前, 别让轮询把选择刷回去
    state.sourcePendingAt = Date.now();     // 押住要有期限, 否则请求丢了就永远不回写
    control({ action: 'source', source: e.target.value });
    state.feedTries = 0;
    state.lastFrame = 0;
    state.lastFrameAt = Date.now();   // 新源从 0 帧开始, 不清掉这个会先误报一次"画面已经停止更新"
    state.paintedAt = 0;
    useStream();
  };
  el('policy').onchange = (e) => control({ action: 'config', policy: e.target.value });
  el('imgsz').onchange = (e) => control({ action: 'config', imgsz: Number(e.target.value) });
  el('strictness').onchange = (e) => control({ action: 'config', min_visibility: Number(e.target.value) });
  const pushSens = debounce((v) => control({ action: 'config', sensitivity: Number(v) }), 250);
  el('sensitivity').oninput = (e) => { el('sens-val').textContent = e.target.value; pushSens(e.target.value); };
  const pushConf = debounce((v) => control({ action: 'config', conf: Number(v) }), 300);
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
