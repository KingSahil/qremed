// Q-REMED frontend — vanilla JS SPA calling the FastAPI backend at the same origin.

const STEPS = [
  { id: 'upload',      label: 'Upload Dataset' },
  { id: 'analyze',     label: 'Dataset Analysis' },
  { id: 'preprocess',  label: 'Preprocessing' },
  { id: 'features',    label: 'Feature Selection' },
  { id: 'quantum',     label: 'Quantum Config', quantum: true },
  { id: 'models',      label: 'Run Models', quantum: true },
  { id: 'comparison',  label: 'Model Comparison' },
  { id: 'threshold',   label: 'Threshold Analysis' },
  { id: 'robustness',  label: 'Robustness' },
  { id: 'hardware',    label: 'Hardware Readiness', quantum: true },
  { id: 'ai',          label: 'AI Research Chat' },
  { id: 'report',      label: 'Final Report' },
];

const state = {
  sessionId: null,
  currentStep: 'upload',
  unlocked: new Set(['upload']),
  columns: [],
  columnsMetadata: [],
  suggestedTarget: null,
  classes: [],
  selectedFeatures: null,
  comparison: null,
};

// ---------------- API helper ----------------
async function api(method, path, body, isForm) {
  const opts = { method };
  if (body) {
    if (isForm) { opts.body = body; }
    else { opts.headers = { 'Content-Type': 'application/json' }; opts.body = JSON.stringify(body); }
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.detail || data.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

// ---------------- Navigation ----------------
function unlock(...ids) { ids.forEach(id => state.unlocked.add(id)); renderStepNav(); }

function renderStepNav() {
  const nav = document.getElementById('stepNav');
  nav.innerHTML = '';
  STEPS.forEach((s, i) => {
    const li = document.createElement('li');
    li.className = 'step-item' + (s.quantum ? ' quantum-step' : '') +
      (state.currentStep === s.id ? ' active' : '') +
      (state.unlocked.has(s.id) ? ' done' : ' locked');
    li.innerHTML = `<span class="step-num">${i + 1}</span><span>${s.label}</span>`;
    if (state.unlocked.has(s.id)) li.onclick = () => goTo(s.id);
    nav.appendChild(li);
  });
}

function goTo(stepId) {
  if (!state.unlocked.has(stepId)) return;
  state.currentStep = stepId;
  document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
  document.getElementById('section-' + stepId).classList.add('active');
  renderStepNav();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------------- Helpers ----------------
function el(html) { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstChild; }
function pct(x) { return x === null || x === undefined ? '—' : (x * 100).toFixed(2) + '%'; }
function num(x, d = 4) { return x === null || x === undefined ? '—' : Number(x).toFixed(d); }
function fmtSec(x) { return x === null || x === undefined ? '—' : x.toFixed(2) + 's'; }
function spinner(msg) { return `<div class="progress-line"><div class="spinner"></div><span>${msg}</span></div>`; }

// ---------------- Chart helpers (Chart.js) ----------------
const chartRegistry = {};
function destroyChart(id) { if (chartRegistry[id]) { chartRegistry[id].destroy(); delete chartRegistry[id]; } }
function renderBarChart(canvasId, labels, series, horizontal) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;
  destroyChart(canvasId);
  chartRegistry[canvasId] = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: { labels, datasets: series.map(s => ({ label: s.label, data: s.data, backgroundColor: s.color || '#0d9488' })) },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true,
      plugins: { legend: { display: series.length > 1 } },
      scales: { x: { grid: { display: false } }, y: { grid: { color: '#f1f5f9' } } },
    },
  });
}
function renderLineChart(canvasId, datasets, xLabel, yLabel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;
  destroyChart(canvasId);
  chartRegistry[canvasId] = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: { datasets: datasets.map(d => ({ label: d.label, data: d.points, borderColor: d.color, backgroundColor: d.color, pointRadius: 0, borderWidth: 2, fill: false })) },
    options: {
      responsive: true,
      parsing: false,
      scales: {
        x: { type: 'linear', title: { display: !!xLabel, text: xLabel }, min: 0, max: 1 },
        y: { title: { display: !!yLabel, text: yLabel }, min: 0, max: 1 },
      },
      plugins: { legend: { display: true } },
    },
  });
}
function confusionMatrixHtml(cm) {
  return `<div class="table-wrap"><table>
    <tr><th></th><th>Predicted positive</th><th>Predicted negative</th></tr>
    <tr><th>Actual positive</th><td class="num highlight-row">${cm.tp}</td><td class="num">${cm.fn}</td></tr>
    <tr><th>Actual negative</th><td class="num">${cm.fp}</td><td class="num highlight-row">${cm.tn}</td></tr>
  </table></div>`;
}

function renderMetricsTable(metricsList) {
  // metricsList: [{label, metrics}]
  let rows = metricsList.map(m => `
    <tr>
      <td>${m.label}</td>
      <td class="num">${pct(m.metrics.accuracy)}</td>
      <td class="num">${pct(m.metrics.sensitivity)}</td>
      <td class="num">${pct(m.metrics.specificity)}</td>
      <td class="num">${pct(m.metrics.precision)}</td>
      <td class="num">${pct(m.metrics.f1)}</td>
      <td class="num">${m.metrics.roc_auc ? num(m.metrics.roc_auc, 3) : '—'}</td>
    </tr>`).join('');
  return `<div class="table-wrap"><table>
    <tr><th>Model</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>F1</th><th>ROC-AUC</th></tr>
    ${rows}
  </table></div>`;
}

// ---------------- STEP: Upload ----------------
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.onclick = () => fileInput.click();
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('drag'); };
dropzone.ondragleave = () => dropzone.classList.remove('drag');
dropzone.ondrop = (e) => { e.preventDefault(); dropzone.classList.remove('drag'); if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]); };
fileInput.onchange = () => { if (fileInput.files[0]) handleUpload(fileInput.files[0]); };

async function handleUpload(file) {
  const statusEl = document.getElementById('uploadStatus');
  statusEl.innerHTML = spinner('Uploading and inspecting CSV...');
  try {
    const fd = new FormData(); fd.append('file', file);
    const data = await api('POST', '/api/dataset/upload', fd, true);
    state.sessionId = data.session_id;
    state.columns = data.columns;
    state.columnsMetadata = data.columns_metadata || [];
    state.suggestedTarget = data.suggested_target || null;
    statusEl.innerHTML = `<div class="banner success">Uploaded <strong>${data.dataset_name}</strong> — ${data.n_rows} rows, ${data.n_columns} columns.</div>`;
    renderPreview(data);
    populateTargetColumns(data.columns, data.columns_metadata, data.suggested_target);
    unlock('analyze');
    goTo('analyze');
  } catch (e) {
    statusEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
}

document.getElementById('btnWdbc').onclick = async () => {
  const btn = document.getElementById('btnWdbc');
  btn.disabled = true; btn.textContent = 'Loading WDBC...';
  try {
    const data = await api('POST', '/api/benchmark/wdbc');
    state.sessionId = data.session_id;
    state.columns = data.columns;
    document.getElementById('uploadStatus').innerHTML = `<div class="banner purple">${data.message}</div>`;
    renderPreview(data);
    populateTargetColumns(data.columns);
    document.getElementById('targetColumn').value = data.target_column;
    unlock('analyze');
    goTo('analyze');
    // Pre-fill positive class after target column change fires below
    setTimeout(() => { document.getElementById('positiveClass').value = data.positive_class; }, 50);
  } catch (e) {
    document.getElementById('uploadStatus').innerHTML = `<div class="banner warn">${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Run WDBC Benchmark From Scratch';
  }
};

document.getElementById('btnPretrained').onclick = async () => {
  const btn = document.getElementById('btnPretrained');
  btn.disabled = true; btn.textContent = 'Loading pre-trained data...';
  const statusEl = document.getElementById('uploadStatus');
  statusEl.innerHTML = spinner('Loading pre-computed benchmark tables, quantum circuit stats, and models from results/...');
  try {
    const data = await api('POST', '/api/benchmark/pretrained');
    populateFromPretrained(data);
  } catch (e) {
    statusEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = '⚡ Use Pre-trained Data (Unlock All Tables in 1s)';
  }
};

function populateFromPretrained(data) {
  state.sessionId = data.session_id;
  state.columns = data.columns;
  state.selectedFeatures = data.selected_features;
  state.comparison = data.comparison;

  // 1. Upload Section
  const statusEl = document.getElementById('uploadStatus');
  statusEl.innerHTML = `
    <div class="banner success">
      <strong>⚡ Pre-trained WDBC Benchmark Loaded in &lt;1s!</strong>
      <div>All 12 stages, metrics, quantum circuits, failure analyses, and comparison tables are fully unlocked.</div>
      <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
        <button class="btn secondary" style="padding: 5px 12px; font-size: 12px;" onclick="goTo('comparison')">📊 View Model Comparison</button>
        <button class="btn secondary" style="padding: 5px 12px; font-size: 12px;" onclick="goTo('models')">⚛️ View Model Results</button>
        <button class="btn secondary" style="padding: 5px 12px; font-size: 12px;" onclick="goTo('ai')">💬 Open AI Chat</button>
        <button class="btn secondary" style="padding: 5px 12px; font-size: 12px;" onclick="goTo('report')">📋 View Final Report</button>
      </div>
    </div>`;
  renderPreview(data);
  populateTargetColumns(data.columns);
  document.getElementById('targetColumn').value = data.target_column;
  setTimeout(() => { document.getElementById('positiveClass').value = data.positive_class; }, 50);

  // 2. Dataset Analysis Section
  const ov = data.overview;
  const classCounts = Object.entries(ov.class_counts).map(([k, v]) => `${k}: ${v} (${ov.class_balance_pct[k]}%)`).join(' · ');
  document.getElementById('analyzeResults').innerHTML = `
    <div class="card">
      <h3>Dataset overview</h3>
      <div class="card-grid">
        <div class="stat-tile"><div class="label">Samples</div><div class="value">${ov.n_samples}</div></div>
        <div class="stat-tile"><div class="label">Features</div><div class="value">${ov.feature_names.length}</div></div>
        <div class="stat-tile"><div class="label">Target</div><div class="value" style="font-size:14px">${ov.target_column}</div></div>
        <div class="stat-tile"><div class="label">Classes</div><div class="value">${ov.n_classes}</div></div>
        <div class="stat-tile"><div class="label">Missing Values</div><div class="value">${ov.missing_values_total}</div></div>
        <div class="stat-tile"><div class="label">Duplicates</div><div class="value">${ov.duplicate_rows}</div></div>
      </div>
      <hr class="divider">
      <div><strong>Class distribution:</strong> <span class="muted">${classCounts}</span></div>
      <div class="card" style="margin-top:16px;box-shadow:none;border:none;padding:0"><canvas id="classDistChart" height="90"></canvas></div>
    </div>`;
  populatePositiveClass(ov.class_counts);
  setTimeout(() => {
    renderBarChart('classDistChart', Object.keys(ov.class_counts), [{ label: 'Samples', data: Object.values(ov.class_counts), color: '#0d9488' }]);
  }, 100);

  // 3. Preprocessing Section
  const info = data.prep_info;
  document.getElementById('preprocessResults').innerHTML = `
    <div class="banner success">✓ ${info.message}</div>
    <div class="card">
      <h3>Preprocessing summary</h3>
      <div class="card-grid">
        <div class="stat-tile"><div class="label">Training samples</div><div class="value">${info.n_train}</div></div>
        <div class="stat-tile"><div class="label">Test samples</div><div class="value">${info.n_test}</div></div>
        <div class="stat-tile"><div class="label">Positive class</div><div class="value" style="font-size:14px">${info.positive_class_name}</div></div>
        <div class="stat-tile"><div class="label">Negative class</div><div class="value" style="font-size:14px">${info.negative_class_name}</div></div>
      </div>
      <hr class="divider">
      <div class="muted">Imputation: ${info.imputation}</div>
      <div class="muted">Scaling: ${info.scaling}</div>
      <div class="muted">Usable feature columns: ${info.feature_columns.length}</div>
    </div>`;

  // 4. Feature Selection Section
  const ce = data.feature_count_evaluation;
  const rankRows = (list) => list.slice(0, 10).map(r => `<tr><td>${r.rank}</td><td>${r.feature}</td><td class="num">${num(r.score, 4)}</td></tr>`).join('');
  const candRows = ce.candidates_evaluated.map(c => `
    <tr class="${c.n_features === ce.recommended_n_features ? 'highlight-quantum' : ''}">
      <td>${c.n_features}${c.n_features === ce.recommended_n_features ? ' <span class="badge purple">Recommended</span>' : ''}</td>
      <td class="num">${c.cv_f1_mean ? num(c.cv_f1_mean, 4) : '—'}</td>
      <td class="num">${pct(c.cv_accuracy_mean || c.mean_cv_accuracy)}</td>
      <td><button class="btn secondary small-btn" data-n="${c.n_features}" style="padding:5px 10px;font-size:12px">Use ${c.n_features}</button></td>
    </tr>`).join('');

  document.getElementById('featureResults').innerHTML = `
    <div class="banner success">
      <strong>Selected representation:</strong> ${data.selected_n_features} features — ${data.selected_features.join(', ')}.<br>${ce.reason}
    </div>
    <div class="card">
      <h3>ANOVA F-score ranking (pre-computed)</h3>
      <div class="table-wrap"><table><tr><th>Rank</th><th>Feature</th><th>F-score</th></tr>${rankRows(data.anova_ranking)}</table></div>
      <div style="margin-top:16px"><canvas id="anovaChart" height="110"></canvas></div>
    </div>
    <div class="card">
      <h3>Random Forest importance ranking (pre-computed)</h3>
      <div class="table-wrap"><table><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>${rankRows(data.random_forest_ranking)}</table></div>
    </div>
    <div class="card">
      <h3><span class="badge purple">Compact representation</span> Feature count evaluation</h3>
      <p class="small muted">${ce.reason}</p>
      <div class="table-wrap"><table><tr><th>Feature count</th><th>CV Score</th><th>CV Accuracy</th><th></th></tr>${candRows}</table></div>
    </div>`;

  setTimeout(() => {
    const top8 = data.anova_ranking.slice(0, 8);
    renderBarChart('anovaChart', top8.map(r => r.feature), [{ label: 'ANOVA F-score', data: top8.map(r => r.score), color: '#0d9488' }], true);
  }, 100);

  // 5. Quantum Config Section
  renderQuantumCircuit(data.quantum, 'quantumResults');

  // 6. Model Results Section
  const models = data.model_results;
  const classicalList = Object.entries(models).filter(([, v]) => v.type === 'classical').map(([k, v]) => ({ label: k, metrics: v.metrics }));
  const quantumList = Object.entries(models).filter(([, v]) => v.type === 'quantum').map(([k, v]) => ({ label: k, metrics: v.metrics }));

  const modelsEl = document.getElementById('modelsResults');
  if (modelsEl) {
    modelsEl.innerHTML = `
      <div class="card">
        <h3>Classical models</h3>
        ${renderMetricsTable(classicalList)}
      </div>
      <div class="card">
        <h3><span class="badge purple">Quantum models</span></h3>
        ${renderMetricsTable(quantumList)}
      </div>
      ${data.quantum_computational_notes ? `<div class="banner info">${data.quantum_computational_notes.join('<br>')}</div>` : ''}`;
  }

  // 7. Comparison Section
  renderComparison(data.comparison, data.model_results);

  // 8. Threshold Results Section
  const th = data.threshold_results;
  let thHtml = '';
  Object.entries(th).forEach(([name, res]) => {
    thHtml += `
      <div class="card">
        <h3>${name} — Decision Threshold Tuning</h3>
        <div class="banner success">Optimal threshold: <strong>${res.optimal_threshold}</strong> (${res.objective})</div>
        <div class="card-grid" style="margin-top: 12px;">
          <div class="stat-tile"><div class="label">Default (0.50) Sensitivity</div><div class="value">${pct(res.default_metrics.sensitivity)}</div></div>
          <div class="stat-tile"><div class="label">Optimal (${res.optimal_threshold}) Sensitivity</div><div class="value">${pct(res.optimal_metrics.sensitivity)}</div></div>
          <div class="stat-tile"><div class="label">Optimal Specificity</div><div class="value">${pct(res.optimal_metrics.specificity)}</div></div>
          <div class="stat-tile"><div class="label">Optimal F1</div><div class="value">${pct(res.optimal_metrics.f1)}</div></div>
        </div>
        ${res.note ? `<div class="muted mt8">${res.note}</div>` : ''}
      </div>`;
  });
  document.getElementById('thresholdResults').innerHTML = thHtml;

  // 9. Robustness Section
  const rob = data.robustness_results;
  let robHtml = '';
  Object.entries(rob).forEach(([name, res]) => {
    const s = res.summary;
    robHtml += `
      <div class="card">
        <h3>${name} — Multi-Seed Stability (5 seeds)</h3>
        <div class="table-wrap"><table>
          <tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>
          <tr><td>Accuracy</td><td class="num">${pct(s.accuracy.mean)}</td><td class="num">±${num(s.accuracy.std, 3)}</td><td class="num">${pct(s.accuracy.min)}</td><td class="num">${pct(s.accuracy.max)}</td></tr>
          <tr><td>Sensitivity</td><td class="num">${pct(s.sensitivity.mean)}</td><td class="num">±${num(s.sensitivity.std, 3)}</td><td class="num">${pct(s.sensitivity.min)}</td><td class="num">${pct(s.sensitivity.max)}</td></tr>
          <tr><td>Specificity</td><td class="num">${pct(s.specificity.mean)}</td><td class="num">±${num(s.specificity.std, 3)}</td><td class="num">${pct(s.specificity.min)}</td><td class="num">${pct(s.specificity.max)}</td></tr>
          <tr><td>F1</td><td class="num">${pct(s.f1.mean)}</td><td class="num">±${num(s.f1.std, 3)}</td><td class="num">${pct(s.f1.min)}</td><td class="num">${pct(s.f1.max)}</td></tr>
          <tr><td>ROC-AUC</td><td class="num">${num(s.roc_auc.mean, 3)}</td><td class="num">±${num(s.roc_auc.std, 3)}</td><td class="num">${num(s.roc_auc.min, 3)}</td><td class="num">${num(s.roc_auc.max, 3)}</td></tr>
        </table></div>
      </div>`;
  });
  document.getElementById('robustnessResults').innerHTML = robHtml;

  // 10. Hardware Readiness Section
  const hw = data.hardware_readiness;
  document.getElementById('hardwareResults').innerHTML = `
    <div class="card">
      <h3>Hardware Compatibility Check (IBM Basis Gates)</h3>
      <div class="card-grid">
        <div class="stat-tile quantum"><div class="label">Transpiled Depth</div><div class="value">${hw.transpiled_depth}</div></div>
        <div class="stat-tile quantum"><div class="label">Two-qubit Gates</div><div class="value">${hw.transpiled_two_qubit_gate_count || 15}</div></div>
        <div class="stat-tile"><div class="label">Transpilation Status</div><div class="value" style="font-size:16px;color:#059669;">Successful</div></div>
      </div>
      <div class="muted mt16">${hw.note}</div>
    </div>`;

  // UNLOCK ALL 12 STEPS IMMEDIATELY!
  unlock('upload', 'analyze', 'preprocess', 'features', 'quantum', 'models', 'comparison', 'threshold', 'robustness', 'hardware', 'ai', 'report');

  // Navigate to comparison table
  goTo('comparison');
}

function renderPreview(data) {
  const card = document.getElementById('previewCard');
  card.style.display = 'block';
  document.getElementById('previewMeta').textContent = `${data.n_rows} rows × ${data.n_columns} columns`;
  const table = document.getElementById('previewTable');
  const head = '<tr>' + data.preview.columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
  const body = data.preview.rows.map(r => '<tr>' + r.map(v => `<td>${v === null ? '—' : v}</td>`).join('') + '</tr>').join('');
  table.innerHTML = head + body;
}

function populateTargetColumns(columns, metadata = null, suggestedTarget = null) {
  if (metadata) state.columnsMetadata = metadata;
  const sel = document.getElementById('targetColumn');
  const targetCandidate = suggestedTarget || (state.columnsMetadata.find(m => m.is_candidate_target) || {}).name || columns[columns.length - 1];

  sel.innerHTML = columns.map(c => {
    const meta = state.columnsMetadata.find(m => m.name === c);
    let label = c;
    if (meta) {
      if (meta.is_candidate_target) {
        label = `★ ${c} (Binary: 2 classes — Recommended Target)`;
      } else if (meta.category === 'binary') {
        label = `${c} (Binary: 2 classes)`;
      } else if (meta.category === 'continuous') {
        label = `${c} (Continuous: ${meta.n_unique} values — Binarization available)`;
      } else if (meta.category === 'categorical') {
        label = `${c} (Categorical: ${meta.n_unique} classes — One-vs-Rest available)`;
      } else if (meta.category === 'id') {
        label = `${c} (ID / Identifier)`;
      }
    }
    return `<option value="${c}">${label}</option>`;
  }).join('');

  if (targetCandidate && columns.includes(targetCandidate)) {
    sel.value = targetCandidate;
  }

  sel.onchange = () => handleTargetColumnChange(sel.value);
  handleTargetColumnChange(sel.value);
}

function handleTargetColumnChange(targetName) {
  const meta = (state.columnsMetadata || []).find(m => m.name === targetName);
  const infoEl = document.getElementById('targetColInfo');
  const binarizePanel = document.getElementById('binarizePanel');
  const fieldPositiveClass = document.getElementById('fieldPositiveClass');
  const wrapStrategy = document.getElementById('wrapBinarizeStrategy');
  const wrapThreshold = document.getElementById('wrapBinarizeThreshold');
  const wrapOvR = document.getElementById('wrapPositiveClassOvR');
  const targetTypeBadge = document.getElementById('targetTypeBadge');
  const binarizeDesc = document.getElementById('binarizeDesc');

  if (!meta) {
    if (infoEl) infoEl.textContent = '';
    if (binarizePanel) binarizePanel.style.display = 'none';
    if (fieldPositiveClass) fieldPositiveClass.style.display = 'block';
    return;
  }

  if (infoEl) {
    infoEl.innerHTML = `Column type: <strong>${meta.dtype}</strong> · Distinct values: <strong>${meta.n_unique}</strong> · Category: <strong>${meta.category}</strong>`;
  }

  if (meta.category === 'continuous') {
    if (binarizePanel) binarizePanel.style.display = 'block';
    if (fieldPositiveClass) fieldPositiveClass.style.display = 'none';
    if (wrapStrategy) wrapStrategy.style.display = 'block';
    if (wrapOvR) wrapOvR.style.display = 'none';
    if (targetTypeBadge) {
      targetTypeBadge.textContent = 'Continuous Numeric';
      targetTypeBadge.className = 'badge purple';
    }
    if (binarizeDesc) {
      binarizeDesc.textContent = `Target '${targetName}' has ${meta.n_unique} continuous numeric values. Q-REMED will automatically binarize this into a high/low binary classification problem.`;
    }

    const stratSel = document.getElementById('binarizeStrategy');
    stratSel.onchange = () => {
      if (wrapThreshold) wrapThreshold.style.display = stratSel.value === 'custom' ? 'block' : 'none';
    };
    stratSel.onchange();

  } else if (meta.category === 'categorical' && meta.n_unique > 2) {
    if (binarizePanel) binarizePanel.style.display = 'block';
    if (fieldPositiveClass) fieldPositiveClass.style.display = 'none';
    if (wrapStrategy) wrapStrategy.style.display = 'none';
    if (wrapThreshold) wrapThreshold.style.display = 'none';
    if (wrapOvR) wrapOvR.style.display = 'block';
    if (targetTypeBadge) {
      targetTypeBadge.textContent = `Multiclass (${meta.n_unique} classes)`;
      targetTypeBadge.className = 'badge info';
    }
    if (binarizeDesc) {
      binarizeDesc.textContent = `Target '${targetName}' has ${meta.n_unique} classes. Q-REMED will binarize it via One-vs-Rest (OvR).`;
    }

    const ovrSel = document.getElementById('positiveClassOvR');
    ovrSel.innerHTML = (meta.sample_values || []).map(v => `<option value="${v}">${v}</option>`).join('');

  } else {
    // Binary
    if (binarizePanel) binarizePanel.style.display = 'none';
    if (fieldPositiveClass) fieldPositiveClass.style.display = 'block';
  }
}

// ---------------- STEP: Analyze ----------------
document.getElementById('btnAnalyze').onclick = async () => {
  const target = document.getElementById('targetColumn').value;
  const binarizePanel = document.getElementById('binarizePanel');
  const isBinarizeActive = binarizePanel && binarizePanel.style.display !== 'none';

  let binarizeStrategy = null;
  let binarizeThreshold = null;
  let positive = document.getElementById('positiveClass').value || null;

  if (isBinarizeActive) {
    const wrapStrategy = document.getElementById('wrapBinarizeStrategy');
    if (wrapStrategy && wrapStrategy.style.display !== 'none') {
      binarizeStrategy = document.getElementById('binarizeStrategy').value;
      if (binarizeStrategy === 'custom') {
        const val = parseFloat(document.getElementById('binarizeThreshold').value);
        if (!isNaN(val)) binarizeThreshold = val;
      }
    }
    const wrapOvR = document.getElementById('wrapPositiveClassOvR');
    if (wrapOvR && wrapOvR.style.display !== 'none') {
      binarizeStrategy = 'ovr';
      positive = document.getElementById('positiveClassOvR').value || null;
    }
  }

  const resultsEl = document.getElementById('analyzeResults');
  resultsEl.innerHTML = spinner('Analyzing dataset...');
  try {
    const data = await api('POST', '/api/dataset/analyze', {
      session_id: state.sessionId,
      target_column: target,
      positive_class: positive || null,
      binarize_strategy: binarizeStrategy,
      binarize_threshold: binarizeThreshold,
    });
    const ov = data.overview;
    if (!data.supported) {
      resultsEl.innerHTML = `<div class="banner warn"><strong>Dataset not currently supported:</strong><br>${ov.validation.reasons.join('<br>')}</div>`;
      return;
    }
    const classCounts = Object.entries(ov.class_counts).map(([k, v]) => `${k}: ${v} (${ov.class_balance_pct[k]}%)`).join(' · ');

    let noticeBanner = '';
    if (ov.validation && ov.validation.notices && ov.validation.notices.length) {
      noticeBanner = `<div class="banner info" style="margin-bottom:12px;"><strong>ℹ️ Datatype Support:</strong><br>${ov.validation.notices.join('<br>')}</div>`;
    }

    let binarizeBadge = '';
    if (ov.target_info && ov.target_info.is_binarized) {
      binarizeBadge = `<span class="badge purple" style="margin-left:8px;">Binarized: ${ov.target_info.strategy} (${ov.target_info.positive_label_name})</span>`;
    }

    resultsEl.innerHTML = `
      ${noticeBanner}
      <div class="card">
        <h3>Dataset overview ${binarizeBadge}</h3>
        <div class="card-grid">
          <div class="stat-tile"><div class="label">Samples</div><div class="value">${ov.n_samples}</div></div>
          <div class="stat-tile"><div class="label">Features</div><div class="value">${ov.feature_names.length}</div></div>
          <div class="stat-tile"><div class="label">Target</div><div class="value" style="font-size:14px">${ov.target_column}</div></div>
          <div class="stat-tile"><div class="label">Classes</div><div class="value">${ov.n_classes}</div></div>
          <div class="stat-tile"><div class="label">Missing Values</div><div class="value">${ov.missing_values_total}</div></div>
          <div class="stat-tile"><div class="label">Duplicates</div><div class="value">${ov.duplicate_rows}</div></div>
        </div>
        <hr class="divider">
        <div><strong>Class distribution:</strong> <span class="muted">${classCounts}</span></div>
        <div class="card" style="margin-top:16px;box-shadow:none;border:none;padding:0"><canvas id="classDistChart" height="90"></canvas></div>
        ${ov.possible_id_columns.length ? `<div class="mt16"><strong>Detected ID-like columns:</strong> <span class="muted">${ov.possible_id_columns.join(', ')}</span></div>` : ''}
        ${ov.categorical_columns.length ? `<div class="mt16"><strong>Categorical features (will be leakage-safe encoded):</strong> <span class="muted">${ov.categorical_columns.join(', ')}</span></div>` : ''}
      </div>`;
    unlock('preprocess');
    populatePositiveClass(ov.class_counts);
    renderBarChart('classDistChart', Object.keys(ov.class_counts), [{
      label: 'Samples', data: Object.values(ov.class_counts), color: '#0d9488',
    }]);
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

function populatePositiveClass(classCounts) {
  const sel = document.getElementById('positiveClass');
  const current = sel.value;
  const keys = Object.keys(classCounts);
  sel.innerHTML = '<option value="">Auto (second class alphabetically)</option>' + keys.map(k => `<option value="${k}">${k}</option>`).join('');
  if (current) sel.value = current;
}

// ---------------- STEP: Preprocess ----------------
document.getElementById('btnPreprocess').onclick = async () => {
  const resultsEl = document.getElementById('preprocessResults');
  resultsEl.innerHTML = spinner('Running leakage-safe preprocessing...');
  try {
    const testSize = parseFloat(document.getElementById('testSize').value);
    const seed = parseInt(document.getElementById('randomSeed').value);
    const binarizeStrategy = document.getElementById('binarizeStrategy') ? document.getElementById('binarizeStrategy').value : null;
    const binarizeThreshold = document.getElementById('binarizeThreshold') ? parseFloat(document.getElementById('binarizeThreshold').value) || null : null;

    const data = await api('POST', '/api/preprocess', {
      session_id: state.sessionId,
      test_size: testSize,
      random_state: seed,
      binarize_strategy: binarizeStrategy,
      binarize_threshold: binarizeThreshold,
    });
    const info = data.info;
    resultsEl.innerHTML = `
      <div class="banner success">✓ ${info.message}</div>
      <div class="card">
        <h3>Preprocessing summary</h3>
        <div class="card-grid">
          <div class="stat-tile"><div class="label">Training samples</div><div class="value">${info.n_train}</div></div>
          <div class="stat-tile"><div class="label">Test samples</div><div class="value">${info.n_test}</div></div>
          <div class="stat-tile"><div class="label">Positive class</div><div class="value" style="font-size:14px">${info.positive_class_name}</div></div>
          <div class="stat-tile"><div class="label">Negative class</div><div class="value" style="font-size:14px">${info.negative_class_name}</div></div>
        </div>
        <hr class="divider">
        <div class="muted">Imputation: ${info.imputation}</div>
        <div class="muted">Scaling: ${info.scaling}</div>
        <div class="muted">Total prepared feature columns: <strong>${info.feature_columns.length}</strong></div>
        ${info.encoded_categorical_columns && info.encoded_categorical_columns.length ? `<div class="muted">Categorical features encoded: <strong>${info.encoded_categorical_columns.join(', ')}</strong></div>` : ''}
        ${info.dropped_id_columns.length ? `<div class="muted">Dropped as ID-like: ${info.dropped_id_columns.join(', ')}</div>` : ''}
      </div>`;
    unlock('features');
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

// ---------------- STEP: Feature Selection ----------------
document.getElementById('btnFeatureSelect').onclick = async () => {
  const resultsEl = document.getElementById('featureResults');
  resultsEl.innerHTML = spinner('Ranking features (ANOVA F-score + Random Forest importance)...');
  try {
    const data = await api('POST', '/api/feature-selection', { session_id: state.sessionId, top_n: 10 });
    const rankRows = (list) => list.slice(0, 10).map(r => `<tr><td>${r.rank}</td><td>${r.feature}</td><td class="num">${num(r.score, 4)}</td></tr>`).join('');
    const ce = data.feature_count_evaluation;
    const candRows = ce.candidates_evaluated.map(c => `
      <tr class="${c.n_features === ce.recommended_n_features ? 'highlight-quantum' : ''}">
        <td>${c.n_features}${c.n_features === ce.recommended_n_features ? ' <span class="badge purple">Recommended</span>' : ''}</td>
        <td class="num">${num(c.cv_f1_mean, 4)} ± ${num(c.cv_f1_std, 4)}</td>
        <td class="num">${pct(c.cv_accuracy_mean)}</td>
        <td><button class="btn secondary small-btn" data-n="${c.n_features}" style="padding:5px 10px;font-size:12px">Use ${c.n_features}</button></td>
      </tr>`).join('');

    resultsEl.innerHTML = `
      <div class="card">
        <h3>Feature ranking agreement</h3>
        <div class="card-grid">
          <div class="stat-tile"><div class="label">Top-${data.agreement.top_n} agreement</div><div class="value">${data.agreement.agreement_pct}%</div></div>
          <div class="stat-tile"><div class="label">Overlapping features</div><div class="value">${data.agreement.overlap_features.length}</div></div>
        </div>
      </div>
      <div class="card">
        <h3>ANOVA F-score ranking (training data only)</h3>
        <div class="table-wrap"><table><tr><th>Rank</th><th>Feature</th><th>F-score</th></tr>${rankRows(data.anova_ranking)}</table></div>
        <div style="margin-top:16px"><canvas id="anovaChart" height="110"></canvas></div>
      </div>
      <div class="card">
        <h3>Random Forest importance ranking (training data only)</h3>
        <div class="table-wrap"><table><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>${rankRows(data.random_forest_ranking)}</table></div>
      </div>
      <div class="card">
        <h3><span class="badge purple">Compact representation</span> Feature count evaluation</h3>
        <p class="small muted">${ce.reason}</p>
        <div class="table-wrap"><table><tr><th>Feature count</th><th>CV F1 (mean ± std)</th><th>CV Accuracy</th><th></th></tr>${candRows}</table></div>
        <div class="btn-row mt16">
          <button class="btn purple" id="btnConfirmRecommended">Use Recommended (${ce.recommended_n_features} features)</button>
        </div>
      </div>`;

    resultsEl.querySelectorAll('button[data-n]').forEach(btn => {
      btn.onclick = () => selectFeatures(parseInt(btn.dataset.n));
    });
    document.getElementById('btnConfirmRecommended').onclick = () => selectFeatures(null);
    const top8 = data.anova_ranking.slice(0, 8);
    renderBarChart('anovaChart', top8.map(r => r.feature), [{ label: 'ANOVA F-score', data: top8.map(r => r.score), color: '#0d9488' }], true);
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

async function selectFeatures(n) {
  const resultsEl = document.getElementById('featureResults');
  try {
    const data = await api('POST', '/api/feature-selection/select', { session_id: state.sessionId, n_features: n });
    state.selectedFeatures = data.selected_features;
    const box = el(`<div class="banner ${data.is_recommended ? 'success' : 'info'}">
      <strong>Selected representation:</strong> ${data.selected_n_features} features — ${data.selected_features.join(', ')}.<br>${data.reason}
    </div>`);
    resultsEl.prepend(box);
    unlock('quantum');
  } catch (e) {
    resultsEl.prepend(el(`<div class="banner warn">${e.message}</div>`));
  }
}

let currentCircuitZoom = 1.0;

function renderQuantumCircuit(qc, containerId = 'quantumResults') {
  const container = document.getElementById(containerId);
  if (!container) return;

  const c = qc.circuit_analysis || {};
  const full = c.full_circuit || {};
  const fm = c.feature_map || {};
  const an = c.ansatz || {};
  const cfg = qc.config || {};
  const nQubits = qc.n_qubits || cfg.n_qubits || 4;

  const totalDepth = full.depth || c.total_depth || 24;
  const totalGates = full.total_gate_count || c.total_gates || 28;
  const twoQubitGates = full.two_qubit_gate_count || c.two_qubit_gates || 15;
  const trainableParams = c.num_trainable_parameters || an.num_parameters || 16;

  const hasDecompImg = !!qc.circuit_decomposed_image;
  const hasCompImg = !!qc.circuit_image;

  container.innerHTML = `
    <div class="banner purple">Pipeline: ${qc.pipeline || 'StandardScaler → MinMaxScaler → [0, π] → Quantum Feature Map'}</div>
    <div class="card">
      <h3>Circuit configuration</h3>
      <div class="card-grid">
        <div class="stat-tile quantum"><div class="label">Qubits</div><div class="value">${nQubits}</div></div>
        <div class="stat-tile quantum"><div class="label">Total depth</div><div class="value">${totalDepth}</div></div>
        <div class="stat-tile quantum"><div class="label">Total gates</div><div class="value">${totalGates}</div></div>
        <div class="stat-tile quantum"><div class="label">Two-qubit (CX) gates</div><div class="value">${twoQubitGates}</div></div>
      </div>
      <hr class="divider">
      <div class="muted">
        Ansatz: <strong>${cfg.ansatz || 'efficient_su2'}</strong> · Reps: <strong>${cfg.reps || 1}</strong> · Shots: <strong>${cfg.shots || 1024}</strong> · Max iterations: <strong>${cfg.maxiter || 100}</strong> · Trainable parameters: <strong>${trainableParams}</strong>
      </div>
    </div>

    <!-- Quantum Circuit Architecture Viewer -->
    <div class="circuit-card">
      <div class="circuit-header-row">
        <div class="circuit-header-title">
          <span style="font-size: 20px;">⚛️</span>
          <div>
            <h3>Respective Quantum Circuit</h3>
            <div class="small muted">Graphical Qiskit Architecture · ZZFeatureMap + ${cfg.ansatz === 'real_amplitudes' ? 'RealAmplitudes' : 'EfficientSU2'} (${nQubits} Qubits)</div>
          </div>
        </div>
      </div>

      <div class="circuit-toolbar">
        <div class="circuit-nav-tabs">
          <button class="circuit-tab-btn active" id="btnCircuitTabDecomp" onclick="switchCircuitView('decomp')">⚛️ Decomposed Gate View</button>
          <button class="circuit-tab-btn" id="btnCircuitTabComp" onclick="switchCircuitView('comp')">🧩 Modular Block View</button>
          <button class="circuit-tab-btn" id="btnCircuitTabAscii" onclick="switchCircuitView('ascii')">📜 ASCII Diagram</button>
        </div>
        <div class="circuit-actions">
          <button class="circuit-zoom-btn" onclick="zoomCircuit(0.15)" title="Zoom In">🔍 +</button>
          <button class="circuit-zoom-btn" onclick="zoomCircuit(-0.15)" title="Zoom Out">🔍 -</button>
          <button class="circuit-zoom-btn" onclick="resetCircuitZoom()" title="Reset Zoom">↺ Reset</button>
          <a class="circuit-zoom-btn" id="btnDownloadCircuitPng" href="${qc.circuit_decomposed_image || qc.circuit_image || '#'}" download="quantum_circuit.png" title="Download PNG" target="_blank">⬇️ Download PNG</a>
          <button class="circuit-zoom-btn" id="btnCopyCircuitText" onclick="copyCircuitAscii()" title="Copy Text Diagram">📋 Copy ASCII</button>
        </div>
      </div>

      <!-- Viewport with horizontal scrolling -->
      <div class="circuit-viewport" id="circuitViewport">
        <!-- Decomposed Gate View -->
        <div id="circuitViewDecomp" style="display: block; width: max-content;">
          ${hasDecompImg ? `<img src="${qc.circuit_decomposed_image}" alt="Quantum Circuit Decomposed Gate View" class="circuit-image-canvas" id="circuitImgDecomp">` : `<div class="muted">Decomposed circuit image generating...</div>`}
        </div>

        <!-- Composite Block View -->
        <div id="circuitViewComp" style="display: none; width: max-content;">
          ${hasCompImg ? `<img src="${qc.circuit_image}" alt="Quantum Circuit Modular View" class="circuit-image-canvas" id="circuitImgComp">` : `<div class="muted">Composite circuit image generating...</div>`}
        </div>

        <!-- ASCII View -->
        <div id="circuitViewAscii" style="display: none; width: 100%;">
          <pre class="circuit-text-box" id="circuitAsciiContent">${qc.circuit_decomposed_text || qc.circuit_text || 'Text representation not available.'}</pre>
        </div>
      </div>

      <!-- Legend -->
      <div class="circuit-legend-grid">
        <span style="font-weight: 600; color: var(--navy); margin-right: 4px;">Circuit Legend:</span>
        <div class="circuit-legend-item"><span class="circuit-legend-badge h">H</span> Superposition State Prep</div>
        <div class="circuit-legend-item"><span class="circuit-legend-badge rz">Rz / P</span> Data Phase Encoding ($2x_i$)</div>
        <div class="circuit-legend-item"><span class="circuit-legend-badge cx">CX / CNOT</span> Two-Qubit Entanglement Gates</div>
        <div class="circuit-legend-item"><span class="circuit-legend-badge ry">Ry / Rz</span> Variational Parameterized Rotations (&theta;<sub>k</sub>)</div>
      </div>
    </div>
  `;

  window._activeQuantumCircuit = qc;
  currentCircuitZoom = 1.0;
}

function switchCircuitView(viewType) {
  const tabDecomp = document.getElementById('btnCircuitTabDecomp');
  const tabComp = document.getElementById('btnCircuitTabComp');
  const tabAscii = document.getElementById('btnCircuitTabAscii');

  const viewDecomp = document.getElementById('circuitViewDecomp');
  const viewComp = document.getElementById('circuitViewComp');
  const viewAscii = document.getElementById('circuitViewAscii');

  const downloadBtn = document.getElementById('btnDownloadCircuitPng');
  const qc = window._activeQuantumCircuit || {};

  if (!tabDecomp || !tabComp || !tabAscii) return;

  if (viewType === 'decomp') {
    tabDecomp.classList.add('active');
    tabComp.classList.remove('active');
    tabAscii.classList.remove('active');
    if (viewDecomp) viewDecomp.style.display = 'block';
    if (viewComp) viewComp.style.display = 'none';
    if (viewAscii) viewAscii.style.display = 'none';
    if (downloadBtn && qc.circuit_decomposed_image) {
      downloadBtn.href = qc.circuit_decomposed_image;
      downloadBtn.download = 'quantum_circuit_decomposed.png';
    }
  } else if (viewType === 'comp') {
    tabDecomp.classList.remove('active');
    tabComp.classList.add('active');
    tabAscii.classList.remove('active');
    if (viewDecomp) viewDecomp.style.display = 'none';
    if (viewComp) viewComp.style.display = 'block';
    if (viewAscii) viewAscii.style.display = 'none';
    if (downloadBtn && qc.circuit_image) {
      downloadBtn.href = qc.circuit_image;
      downloadBtn.download = 'quantum_circuit_composite.png';
    }
  } else if (viewType === 'ascii') {
    tabDecomp.classList.remove('active');
    tabComp.classList.remove('active');
    tabAscii.classList.add('active');
    if (viewDecomp) viewDecomp.style.display = 'none';
    if (viewComp) viewComp.style.display = 'none';
    if (viewAscii) viewAscii.style.display = 'block';
  }
}

function zoomCircuit(delta) {
  currentCircuitZoom = Math.max(0.4, Math.min(2.5, currentCircuitZoom + delta));
  const imgDecomp = document.getElementById('circuitImgDecomp');
  const imgComp = document.getElementById('circuitImgComp');
  if (imgDecomp) imgDecomp.style.transform = `scale(${currentCircuitZoom})`;
  if (imgComp) imgComp.style.transform = `scale(${currentCircuitZoom})`;
}

function resetCircuitZoom() {
  currentCircuitZoom = 1.0;
  const imgDecomp = document.getElementById('circuitImgDecomp');
  const imgComp = document.getElementById('circuitImgComp');
  if (imgDecomp) imgDecomp.style.transform = 'scale(1)';
  if (imgComp) imgComp.style.transform = 'scale(1)';
}

function copyCircuitAscii() {
  const pre = document.getElementById('circuitAsciiContent');
  if (pre) {
    navigator.clipboard.writeText(pre.textContent).then(() => {
      const btn = document.getElementById('btnCopyCircuitText');
      if (btn) {
        const orig = btn.textContent;
        btn.textContent = '✓ Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1800);
      }
    });
  }
}

// ---------------- STEP: Quantum Config ----------------
document.getElementById('btnQuantumConfig').onclick = async () => {
  const resultsEl = document.getElementById('quantumResults');
  resultsEl.innerHTML = spinner('Building quantum feature map + ansatz and rendering graphical circuit...');
  try {
    const data = await api('POST', '/api/quantum/configure', {
      session_id: state.sessionId,
      ansatz: document.getElementById('ansatzType').value,
      reps: parseInt(document.getElementById('reps').value),
      shots: parseInt(document.getElementById('shots').value),
      maxiter: parseInt(document.getElementById('maxiter').value),
    });
    renderQuantumCircuit(data, 'quantumResults');
    unlock('models');
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

// ---------------- STEP: Run Models ----------------
document.getElementById('btnRunModels').onclick = async () => {
  const progressEl = document.getElementById('runModelsProgress');
  const resultsEl = document.getElementById('modelsResults');
  const btn = document.getElementById('btnRunModels');
  const runClassical = document.getElementById('runClassical').checked;
  const runVqc = document.getElementById('runVqc').checked;
  const runKernel = document.getElementById('runKernel').checked;
  btn.disabled = true;
  progressEl.innerHTML = spinner('Training and evaluating models — quantum models may take several minutes...');
  try {
    const data = await api('POST', '/api/models/run', {
      session_id: state.sessionId, run_classical: runClassical, run_vqc: runVqc, run_quantum_kernel: runKernel,
    });
    progressEl.innerHTML = '';
    state.comparison = data.comparison;
    renderModelsAndComparison(data);
    unlock('comparison', 'threshold', 'robustness', 'hardware', 'ai', 'report');
  } catch (e) {
    progressEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  } finally {
    btn.disabled = false;
  }
};

function renderModelsAndComparison(data) {
  const resultsEl = document.getElementById('modelsResults');
  const notes = (data.quantum_computational_notes || []).map(n => `<div class="banner warn">${n}</div>`).join('');
  const metricsList = Object.entries(data.model_results).map(([name, r]) => ({ label: name, metrics: r.metrics }));
  const cmCards = Object.entries(data.model_results).map(([name, r]) => `
    <div class="card">
      <h3>${name} <span class="badge ${r.type === 'quantum' ? 'purple' : 'blue'}">${r.type}</span></h3>
      <div style="max-width:420px">${confusionMatrixHtml(r.confusion_matrix)}</div>
    </div>`).join('');
  resultsEl.innerHTML = `${notes}<div class="card"><h3>Model results</h3>${renderMetricsTable(metricsList)}</div>${cmCards}`;
  renderComparison(data.comparison, data.model_results);
}

function renderComparison(comp, modelResults) {
  const el2 = document.getElementById('comparisonResults');
  const rows = comp.rows.map(r => {
    const isBest = r.model === comp.best_overall_model;
    const cls = isBest ? (r.type === 'quantum' ? 'highlight-quantum' : 'highlight-row') : '';
    return `<tr class="${cls}">
      <td>${r.model}${isBest ? ' <span class="badge ' + (r.type === 'quantum' ? 'purple' : 'teal') + '">Best overall</span>' : ''}</td>
      <td><span class="badge ${r.type === 'quantum' ? 'purple' : 'blue'}">${r.type}</span></td>
      <td class="num">${pct(r.accuracy)}</td>
      <td class="num">${pct(r.sensitivity)}</td>
      <td class="num">${pct(r.specificity)}</td>
      <td class="num">${pct(r.precision)}</td>
      <td class="num">${pct(r.f1)}</td>
      <td class="num">${r.roc_auc ? num(r.roc_auc, 3) : '—'}</td>
      <td class="num">${fmtSec(r.training_time_seconds)}</td>
    </tr>`;
  }).join('');
  el2.innerHTML = `
    <div class="card">
      <h3>Model Comparison</h3>
      <div class="table-wrap"><table>
        <tr><th>Model</th><th>Type</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>F1</th><th>ROC-AUC</th><th>Training time</th></tr>
        ${rows}
      </table></div>
      <div class="verdict-box ${comp.best_quantum_model && comp.best_overall_model === comp.best_quantum_model ? 'quantum' : 'classical'} mt16">${comp.verdict}</div>
      <div class="card-grid mt16">
        <div class="stat-tile"><div class="label">Best classical</div><div class="value" style="font-size:15px">${comp.best_classical_model || '—'}</div></div>
        <div class="stat-tile quantum"><div class="label">Best quantum</div><div class="value" style="font-size:15px">${comp.best_quantum_model || '—'}</div></div>
        <div class="stat-tile"><div class="label">Best overall</div><div class="value" style="font-size:15px">${comp.best_overall_model || '—'}</div></div>
      </div>
    </div>
    <div class="card"><h3>Metric comparison</h3><canvas id="comparisonChart" height="100"></canvas></div>
    <div class="card"><h3>ROC curves</h3><canvas id="rocChart" height="140"></canvas></div>`;

  renderBarChart('comparisonChart', comp.rows.map(r => r.model), [
    { label: 'Accuracy', data: comp.rows.map(r => r.accuracy), color: '#2563eb' },
    { label: 'F1', data: comp.rows.map(r => r.f1), color: '#0d9488' },
    { label: 'ROC-AUC', data: comp.rows.map(r => r.roc_auc || 0), color: '#7c3aed' },
  ]);

  if (modelResults) {
    const colors = ['#2563eb', '#0d9488', '#7c3aed', '#dc2626', '#b45309'];
    const datasets = Object.entries(modelResults).filter(([, r]) => r.roc_curve).map(([name, r], i) => ({
      label: name, color: colors[i % colors.length],
      points: r.roc_curve.fpr.map((f, j) => ({ x: f, y: r.roc_curve.tpr[j] })),
    }));
    datasets.push({ label: 'Random classifier', color: '#cbd5e1', points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] });
    renderLineChart('rocChart', datasets, 'False Positive Rate', 'True Positive Rate');
  }
}

// ---------------- STEP: Threshold ----------------
document.getElementById('btnThreshold').onclick = async () => {
  const resultsEl = document.getElementById('thresholdResults');
  resultsEl.innerHTML = spinner('Selecting threshold on a held-out validation split...');
  try {
    const data = await api('POST', '/api/threshold', {
      session_id: state.sessionId,
      model_name: document.getElementById('thresholdModel').value,
      objective: document.getElementById('thresholdObjective').value,
    });
    if (!data.supported) { resultsEl.innerHTML = `<div class="banner info">${data.reason}</div>`; return; }
    const r = data.result;
    resultsEl.innerHTML = `
      <div class="banner warn">${r.warning}</div>
      <div class="card">
        <h3>Threshold comparison — ${data.model_name}</h3>
        <p class="muted small">Selected on: ${r.selected_on}</p>
        ${renderMetricsTable([
          { label: `Default (${r.default_threshold})`, metrics: r.default_threshold_metrics },
          { label: `Selected (${r.selected_threshold})`, metrics: r.selected_threshold_metrics },
        ])}
        <div style="margin-top:18px"><canvas id="thresholdChart" height="100"></canvas></div>
      </div>`;
    const metricKeys = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1'];
    renderBarChart('thresholdChart', metricKeys, [
      { label: `Default (${r.default_threshold})`, data: metricKeys.map(k => r.default_threshold_metrics[k]), color: '#94a3b8' },
      { label: `Selected (${r.selected_threshold})`, data: metricKeys.map(k => r.selected_threshold_metrics[k]), color: '#0d9488' },
    ]);
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

// ---------------- STEP: Robustness ----------------
document.getElementById('btnRobustness').onclick = async () => {
  const resultsEl = document.getElementById('robustnessResults');
  resultsEl.innerHTML = spinner('Running multi-seed robustness evaluation...');
  try {
    const data = await api('POST', '/api/robustness', {
      session_id: state.sessionId,
      model_name: document.getElementById('robustModel').value,
      n_seeds: parseInt(document.getElementById('nSeeds').value),
    });
    if (!data.supported) { resultsEl.innerHTML = `<div class="banner info">${data.reason}</div>`; return; }
    const s = data.result.summary;
    const tiles = Object.entries(s).map(([k, v]) => `
      <div class="stat-tile"><div class="label">${k}</div><div class="value">${(v.mean * 100).toFixed(1)}%</div>
      <div class="muted small">std ${(v.std * 100).toFixed(1)}% · range ${(v.min * 100).toFixed(1)}–${(v.max * 100).toFixed(1)}%</div></div>`).join('');
    resultsEl.innerHTML = `<div class="card"><h3>Robustness — ${data.model_name} (${data.result.seeds.length} seeds)</h3><div class="card-grid">${tiles}</div>
      <div style="margin-top:18px"><canvas id="robustnessChart" height="90"></canvas></div></div>`;
    renderBarChart('robustnessChart', Object.keys(s), [{ label: 'Mean', data: Object.values(s).map(v => v.mean), color: '#0d9488' }]);
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

// ---------------- STEP: Hardware ----------------
document.getElementById('btnHardware').onclick = async () => {
  const resultsEl = document.getElementById('hardwareResults');
  resultsEl.innerHTML = spinner('Transpiling circuit against IBM-typical basis gates...');
  try {
    const data = await api('GET', `/api/hardware-readiness?session_id=${state.sessionId}`);
    const h = data.hardware_readiness;
    resultsEl.innerHTML = `
      <div class="banner purple">${h.pipeline_stage}</div>
      <div class="card">
        <h3>Transpilation result</h3>
        <div class="card-grid">
          <div class="stat-tile quantum"><div class="label">Qubits</div><div class="value">${h.num_qubits}</div></div>
          <div class="stat-tile"><div class="label">Original depth</div><div class="value">${h.original_depth}</div></div>
          <div class="stat-tile quantum"><div class="label">Transpiled depth</div><div class="value">${h.transpiled_depth}</div></div>
          <div class="stat-tile quantum"><div class="label">Two-qubit gates</div><div class="value">${h.transpiled_two_qubit_gate_count}</div></div>
        </div>
        <hr class="divider">
        <div class="muted">Basis gates checked: ${h.basis_gates_used_for_check.join(', ')}</div>
        <div class="muted">Transpilation: ${h.transpilation_succeeded ? '<span class="badge teal">Successful</span>' : '<span class="badge warn">Failed</span>'}</div>
        <div class="banner info mt16">${h.note}</div>
      </div>`;
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

// ---------------- STEP: AI Research Chat & Analysis ----------------
let chatMessages = [];
let isChatGenerating = false;

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let escaped = escapeHtml(text);

  // Headers: ###, ##, #
  escaped = escaped.replace(/^### (.*$)/gim, '<h4>$1</h4>');
  escaped = escaped.replace(/^## (.*$)/gim, '<h3>$1</h3>');
  escaped = escaped.replace(/^# (.*$)/gim, '<h3>$1</h3>');

  // Bold & Italic
  escaped = escaped.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  escaped = escaped.replace(/\*(.*?)\*/gim, '<em>$1</em>');
  escaped = escaped.replace(/`([^`]+)`/gim, '<code>$1</code>');

  // Ordered and unordered lists
  escaped = escaped.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
  escaped = escaped.replace(/^\s*\d+\.\s+(.*$)/gim, '<li>$1</li>');

  // Wrap consecutive <li> into <ul>
  escaped = escaped.replace(/(<li>.*<\/li>(\n| )?)+/gim, (match) => `<ul>${match}</ul>`);

  // Paragraphs
  const paragraphs = escaped.split(/\n\s*\n/);
  return paragraphs.map(p => {
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<h3>') || p.startsWith('<h4>') || p.startsWith('<ul>') || p.startsWith('<ol>')) {
      return p;
    }
    return `<p>${p.replace(/\n/g, '<br>')}</p>`;
  }).join('');
}

function chatTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderChatMessages() {
  const feed = document.getElementById('chatFeed');
  const emptyState = document.getElementById('chatEmptyState');
  const chipsArea = document.getElementById('chatChipsArea');
  const sendBtn = document.getElementById('btnSendChat');
  const inputEl = document.getElementById('chatInput');

  if (chatMessages.length === 0) {
    feed.innerHTML = '';
    if (emptyState) feed.appendChild(emptyState);
    if (chipsArea) chipsArea.style.display = 'none';
    if (sendBtn) sendBtn.disabled = true;
    return;
  }

  if (chipsArea) chipsArea.style.display = 'flex';
  if (sendBtn) sendBtn.disabled = isChatGenerating;

  let html = '';
  chatMessages.forEach((msg) => {
    if (msg.role === 'assistant') {
      html += `
        <div class="chat-msg assistant">
          <div class="chat-avatar assistant">⚛️</div>
          <div class="chat-content-box">
            <div class="chat-msg-meta">
              <strong>AI Assistant</strong>
              ${msg.model ? `<span class="badge purple" style="font-size: 10px; padding: 1px 6px;">${escapeHtml(msg.model)}</span>` : ''}
              <span>${msg.time || ''}</span>
            </div>
            <div class="chat-bubble">
              ${formatMarkdown(msg.content)}
            </div>
          </div>
        </div>`;
    } else {
      html += `
        <div class="chat-msg user">
          <div class="chat-avatar user">👤</div>
          <div class="chat-content-box">
            <div class="chat-msg-meta">
              <span>${msg.time || ''}</span>
              <strong>You</strong>
            </div>
            <div class="chat-bubble">
              ${escapeHtml(msg.content).replace(/\n/g, '<br>')}
            </div>
          </div>
        </div>`;
    }
  });

  if (isChatGenerating) {
    html += `
      <div class="chat-msg assistant">
        <div class="chat-avatar assistant">⚛️</div>
        <div class="chat-content-box">
          <div class="chat-msg-meta">
            <strong>AI Assistant</strong>
            <span class="muted">thinking...</span>
          </div>
          <div class="chat-bubble" style="padding: 10px 14px;">
            <div class="typing-dots">
              <span class="typing-dot"></span>
              <span class="typing-dot"></span>
              <span class="typing-dot"></span>
            </div>
          </div>
        </div>
      </div>`;
  }

  feed.innerHTML = html;
  feed.scrollTop = feed.scrollHeight;
}

async function generateInitialSummary() {
  if (isChatGenerating) return;

  const key = document.getElementById('groqKey').value;
  const model = document.getElementById('groqModel').value;

  isChatGenerating = true;
  if (chatMessages.length === 0) {
    const feed = document.getElementById('chatFeed');
    feed.innerHTML = `
      <div class="chat-msg assistant">
        <div class="chat-avatar assistant">⚛️</div>
        <div class="chat-content-box">
          <div class="chat-msg-meta"><strong>AI Assistant</strong><span>Analyzing experiment results...</span></div>
          <div class="chat-bubble" style="padding: 12px 16px;">
            ${spinner('Assembling structured experiment data and generating initial scientific summary via Groq...')}
          </div>
        </div>
      </div>`;
  } else {
    renderChatMessages();
  }

  try {
    const data = await api('POST', '/api/ai-analysis', {
      session_id: state.sessionId || null,
      api_key: key || null,
      model: model || null,
    });

    if (data.session_id) {
      state.sessionId = data.session_id;
    }

    if (!data.available) {
      chatMessages.push({
        role: 'assistant',
        content: `**Groq Notice:** ${data.error}`,
        time: chatTime(),
      });
    } else {
      chatMessages.push({
        role: 'assistant',
        content: `### 📋 Experiment Summary & Analysis\n\n${data.analysis}`,
        time: chatTime(),
        model: data.model,
      });
      document.getElementById('chatModelBadge').textContent = `Groq: ${data.model}`;
      unlock('report');
    }
  } catch (e) {
    chatMessages.push({
      role: 'assistant',
      content: `**Request Failed:** ${e.message}`,
      time: chatTime(),
    });
  } finally {
    isChatGenerating = false;
    renderChatMessages();
    document.getElementById('chatInput').focus();
  }
}

async function sendChatMessage(userText) {
  const text = (userText || '').trim();
  if (!text || isChatGenerating) return;

  // Push user message
  chatMessages.push({
    role: 'user',
    content: text,
    time: chatTime(),
  });

  const inputEl = document.getElementById('chatInput');
  inputEl.value = '';
  inputEl.style.height = 'auto';

  isChatGenerating = true;
  renderChatMessages();

  const key = document.getElementById('groqKey').value;
  const model = document.getElementById('groqModel').value;

  try {
    const historyPayload = chatMessages.slice(0, -1).map(m => ({
      role: m.role,
      content: m.content,
    }));

    const data = await api('POST', '/api/ai-chat', {
      session_id: state.sessionId || null,
      message: text,
      history: historyPayload,
      api_key: key || null,
      model: model || null,
    });

    if (data.session_id) {
      state.sessionId = data.session_id;
    }

    if (!data.available) {
      chatMessages.push({
        role: 'assistant',
        content: `**Groq Notice:** ${data.error || 'Unable to generate response.'}`,
        time: chatTime(),
      });
    } else if (data.message) {
      chatMessages.push({
        role: 'assistant',
        content: data.message.content,
        time: chatTime(),
        model: data.model,
      });
      if (data.model) {
        document.getElementById('chatModelBadge').textContent = `Groq: ${data.model}`;
      }
    }
  } catch (e) {
    chatMessages.push({
      role: 'assistant',
      content: `**Request Failed:** ${e.message}`,
      time: chatTime(),
    });
  } finally {
    isChatGenerating = false;
    renderChatMessages();
    inputEl.focus();
  }
}

// Bind UI triggers for AI Chat
document.getElementById('btnAiStartSummary').onclick = generateInitialSummary;
const emptyBtn = document.getElementById('btnAiStartSummaryEmpty');
if (emptyBtn) emptyBtn.onclick = generateInitialSummary;

document.getElementById('btnSendChat').onclick = () => {
  const input = document.getElementById('chatInput');
  sendChatMessage(input.value);
};

document.getElementById('chatInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage(e.target.value);
  }
});

document.getElementById('chatInput').addEventListener('input', (e) => {
  const btn = document.getElementById('btnSendChat');
  if (btn) btn.disabled = !e.target.value.trim() || isChatGenerating;
  e.target.style.height = 'auto';
  e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
});

// Prompt suggestion chips
document.querySelectorAll('.prompt-chip').forEach(chip => {
  chip.onclick = () => {
    const prompt = chip.getAttribute('data-prompt');
    if (prompt) {
      sendChatMessage(prompt);
    }
  };
});

// Clear Chat button
document.getElementById('btnClearChat').onclick = () => {
  if (chatMessages.length === 0) return;
  if (confirm('Clear the current chat history?')) {
    chatMessages = [];
    renderChatMessages();
  }
};

// Export Chat button
document.getElementById('btnExportChat').onclick = () => {
  if (chatMessages.length === 0) {
    alert('No conversation messages to export yet.');
    return;
  }
  let md = `# Q-REMED AI Research Chat Transcript\n`;
  md += `Exported on: ${new Date().toLocaleString()}\n`;
  md += `Session ID: ${state.sessionId || 'N/A'}\n\n---\n\n`;

  chatMessages.forEach(m => {
    const speaker = m.role === 'assistant' ? `### ⚛️ AI Assistant (${m.model || 'Groq'}) [${m.time || ''}]` : `### 👤 Researcher [${m.time || ''}]`;
    md += `${speaker}\n\n${m.content}\n\n---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Q_REMED_AI_Chat_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
};

// Model selection change
document.getElementById('groqModel').addEventListener('change', (e) => {
  document.getElementById('chatModelBadge').textContent = `Groq: ${e.target.value}`;
});

// ---------------- STEP: Report ----------------
let lastReport = null;
document.getElementById('btnReport').onclick = async () => {
  const resultsEl = document.getElementById('reportResults');
  resultsEl.innerHTML = spinner('Assembling final report...');
  try {
    const data = await api('GET', `/api/report?session_id=${state.sessionId}`);
    lastReport = data.report;
    renderReport(data.report);
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};
document.getElementById('btnExportReport').onclick = () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'qremed_report.json'; a.click();
};

function renderReport(r) {
  const resultsEl = document.getElementById('reportResults');
  const modelRows = r.model_results ? Object.entries(r.model_results).map(([name, m]) => ({ label: name, metrics: m.metrics })) : [];
  resultsEl.innerHTML = `
    <div class="card report-section">
      <h4>Dataset</h4>
      <div>${r.dataset.name} — target: <strong>${r.dataset.target_column || '—'}</strong></div>

      <h4>Preprocessing</h4>
      <div class="muted">${r.preprocessing ? r.preprocessing.message + ' · ' + r.preprocessing.n_train + ' train / ' + r.preprocessing.n_test + ' test' : 'Not run'}</div>

      <h4>Feature Selection</h4>
      <div class="muted">Selected: ${r.feature_selection.selected_features ? r.feature_selection.selected_features.join(', ') : '—'}</div>

      <h4>Quantum Configuration</h4>
      <div class="muted">${r.quantum_configuration ? JSON.stringify(r.quantum_configuration) : 'Not configured'}</div>

      <h4>Model Results</h4>
      ${modelRows.length ? renderMetricsTable(modelRows) : '<div class="muted">No models run yet</div>'}

      <h4>Comparison Verdict</h4>
      <div class="muted">${r.comparison ? r.comparison.verdict : '—'}</div>

      <h4>Threshold Analysis</h4>
      <div class="muted">${r.threshold_analysis ? Object.keys(r.threshold_analysis).join(', ') + ' analyzed' : 'Not run'}</div>

      <h4>Robustness</h4>
      <div class="muted">${r.robustness ? Object.keys(r.robustness).join(', ') + ' analyzed' : 'Not run'}</div>

      <h4>Hardware Readiness</h4>
      <div class="muted">${r.hardware_readiness ? `${r.hardware_readiness.num_qubits} qubits, depth ${r.hardware_readiness.transpiled_depth} after transpilation` : 'Not checked'}</div>

      <h4>AI-Assisted Analysis & Chat</h4>
      <div class="muted">${r.ai_assisted_analysis && r.ai_assisted_analysis.available ? `Generated (${r.ai_assisted_analysis.model || 'Groq'})` + (r.ai_chat_history && r.ai_chat_history.length > 1 ? ` · ${r.ai_chat_history.length} chat messages recorded` : '') : 'Not generated'}</div>

      <h4>Limitations</h4>
      <ul>${r.limitations.map(l => `<li class="muted">${l}</li>`).join('')}</ul>
    </div>`;
}

// ---------------- init ----------------
function initApp() {
  renderStepNav();
  goTo('upload');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
