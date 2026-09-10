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
  { id: 'ai',          label: 'AI-Assisted Analysis' },
  { id: 'report',      label: 'Final Report' },
];

const state = {
  sessionId: null,
  currentStep: 'upload',
  unlocked: new Set(['upload']),
  columns: [],
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
    statusEl.innerHTML = `<div class="banner success">Uploaded <strong>${data.dataset_name}</strong> — ${data.n_rows} rows, ${data.n_columns} columns.</div>`;
    renderPreview(data);
    populateTargetColumns(data.columns);
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
    btn.disabled = false; btn.textContent = 'Run WDBC Benchmark';
  }
};

function renderPreview(data) {
  const card = document.getElementById('previewCard');
  card.style.display = 'block';
  document.getElementById('previewMeta').textContent = `${data.n_rows} rows × ${data.n_columns} columns`;
  const table = document.getElementById('previewTable');
  const head = '<tr>' + data.preview.columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
  const body = data.preview.rows.map(r => '<tr>' + r.map(v => `<td>${v === null ? '—' : v}</td>`).join('') + '</tr>').join('');
  table.innerHTML = head + body;
}

function populateTargetColumns(columns) {
  const sel = document.getElementById('targetColumn');
  sel.innerHTML = columns.map(c => `<option value="${c}">${c}</option>`).join('');
  sel.onchange = () => { document.getElementById('positiveClass').innerHTML = '<option value="">Auto (second class alphabetically)</option>'; };
}

// ---------------- STEP: Analyze ----------------
document.getElementById('btnAnalyze').onclick = async () => {
  const target = document.getElementById('targetColumn').value;
  const positive = document.getElementById('positiveClass').value;
  const resultsEl = document.getElementById('analyzeResults');
  resultsEl.innerHTML = spinner('Analyzing dataset...');
  try {
    const data = await api('POST', '/api/dataset/analyze', { session_id: state.sessionId, target_column: target, positive_class: positive || null });
    const ov = data.overview;
    if (!data.supported) {
      resultsEl.innerHTML = `<div class="banner warn"><strong>Dataset not currently supported:</strong><br>${ov.validation.reasons.join('<br>')}</div>`;
      return;
    }
    const classCounts = Object.entries(ov.class_counts).map(([k, v]) => `${k}: ${v} (${ov.class_balance_pct[k]}%)`).join(' · ');
    resultsEl.innerHTML = `
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
        ${ov.possible_id_columns.length ? `<div class="mt16"><strong>Detected ID-like columns:</strong> <span class="muted">${ov.possible_id_columns.join(', ')}</span></div>` : ''}
        ${ov.categorical_columns.length ? `<div class="mt16"><strong>Categorical columns (will be excluded):</strong> <span class="muted">${ov.categorical_columns.join(', ')}</span></div>` : ''}
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
    const data = await api('POST', '/api/preprocess', { session_id: state.sessionId, test_size: testSize, random_state: seed });
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
        <div class="muted">Usable feature columns: ${info.feature_columns.length}</div>
        ${info.dropped_id_columns.length ? `<div class="muted">Dropped as ID-like: ${info.dropped_id_columns.join(', ')}</div>` : ''}
        ${info.dropped_categorical_columns.length ? `<div class="muted">Dropped as categorical: ${info.dropped_categorical_columns.join(', ')}</div>` : ''}
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

// ---------------- STEP: Quantum Config ----------------
document.getElementById('btnQuantumConfig').onclick = async () => {
  const resultsEl = document.getElementById('quantumResults');
  resultsEl.innerHTML = spinner('Building quantum feature map + ansatz...');
  try {
    const data = await api('POST', '/api/quantum/configure', {
      session_id: state.sessionId,
      ansatz: document.getElementById('ansatzType').value,
      reps: parseInt(document.getElementById('reps').value),
      shots: parseInt(document.getElementById('shots').value),
      maxiter: parseInt(document.getElementById('maxiter').value),
    });
    const c = data.circuit_analysis;
    resultsEl.innerHTML = `
      <div class="banner purple">Pipeline: ${data.pipeline}</div>
      <div class="card">
        <h3>Circuit configuration</h3>
        <div class="card-grid">
          <div class="stat-tile quantum"><div class="label">Qubits</div><div class="value">${data.n_qubits}</div></div>
          <div class="stat-tile quantum"><div class="label">Circuit depth</div><div class="value">${c.full_circuit.depth}</div></div>
          <div class="stat-tile quantum"><div class="label">Two-qubit gates</div><div class="value">${c.full_circuit.two_qubit_gate_count}</div></div>
          <div class="stat-tile quantum"><div class="label">Trainable params</div><div class="value">${c.num_trainable_parameters}</div></div>
        </div>
        <hr class="divider">
        <div class="muted">Feature map: ZZFeatureMap — depth ${c.feature_map.depth}, ${c.feature_map.num_parameters} input parameters</div>
        <div class="muted">Ansatz: ${data.config.ansatz} — depth ${c.ansatz.depth}, ${c.ansatz.num_parameters} trainable parameters</div>
      </div>`;
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

// ---------------- STEP: AI Analysis ----------------
document.getElementById('btnAiAnalysis').onclick = async () => {
  const resultsEl = document.getElementById('aiResults');
  resultsEl.innerHTML = spinner('Requesting Groq analysis of the structured results...');
  try {
    const key = document.getElementById('groqKey').value;
    const data = await api('POST', '/api/ai-analysis', { session_id: state.sessionId, api_key: key || null });
    if (!data.available) { resultsEl.innerHTML = `<div class="banner info">${data.error}</div>`; return; }
    resultsEl.innerHTML = `
      <div class="card">
        <h3><span class="badge purple">${data.label}</span></h3>
        <div class="ai-output">${data.analysis.replace(/</g, '&lt;')}</div>
      </div>`;
  } catch (e) {
    resultsEl.innerHTML = `<div class="banner warn">${e.message}</div>`;
  }
};

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

      <h4>AI-Assisted Analysis</h4>
      <div class="muted">${r.ai_assisted_analysis && r.ai_assisted_analysis.available ? 'Generated' : 'Not generated'}</div>

      <h4>Limitations</h4>
      <ul>${r.limitations.map(l => `<li class="muted">${l}</li>`).join('')}</ul>
    </div>`;
}

// ---------------- init ----------------
renderStepNav();
goTo('upload');
