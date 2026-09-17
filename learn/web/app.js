/**
 * GNN Advanced Systems & Empirical Verification: app.js
 * Part 2: Builds on foundations from GNN_Basics.html.
 * Renders Zachary Karate Club benchmark, scrollbar-free matrix heatmap, GAT attention, and RTX 5050 runs.
 */

// ── Real Experiment Datasets (RTX 5050 Gesture & Sign Experiments) ───────────
const STATIC_GCN_DATA = {
  "device": "cuda:0",
  "device_name": "NVIDIA GeForce RTX 5050 Laptop GPU",
  "total_samples": 300,
  "train_samples": 240,
  "test_samples": 60,
  "epochs": 10,
  "final_test_accuracy": 86.67,
  "duration_seconds": 1.04,
  "history": [
    { "epoch": 1, "train_loss": 2.0611, "train_acc": 21.67, "test_acc": 10.0 },
    { "epoch": 2, "train_loss": 1.6659, "train_acc": 42.50, "test_acc": 10.0 },
    { "epoch": 3, "train_loss": 1.4244, "train_acc": 55.00, "test_acc": 10.0 },
    { "epoch": 4, "train_loss": 1.2726, "train_acc": 65.83, "test_acc": 10.0 },
    { "epoch": 5, "train_loss": 1.1165, "train_acc": 68.33, "test_acc": 71.67 },
    { "epoch": 6, "train_loss": 0.9458, "train_acc": 70.83, "test_acc": 45.00 },
    { "epoch": 7, "train_loss": 0.8693, "train_acc": 72.92, "test_acc": 68.33 },
    { "epoch": 8, "train_loss": 0.8048, "train_acc": 75.83, "test_acc": 30.00 },
    { "epoch": 9, "train_loss": 0.7582, "train_acc": 73.75, "test_acc": 51.67 },
    { "epoch": 10, "train_loss": 0.7288, "train_acc": 72.50, "test_acc": 86.67 }
  ],
  "classes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
};

const STGCN_DATA = {
  "device": "cuda:0",
  "device_name": "NVIDIA GeForce RTX 5050 Laptop GPU",
  "dataset": "Simulated How2Sign + iSign Subsets",
  "sequence_shape": "[B, C=3, T=30, V=21]",
  "num_gloss_classes": 8,
  "glosses": ["HELLO", "THANK_YOU", "PLEASE", "HELP", "YES", "NO", "LEARN", "NAME"],
  "epochs": 10,
  "final_val_loss": 0.1683,
  "final_val_accuracy": 100.0,
  "duration_seconds": 1.04,
  "history": [
    { "epoch": 1, "train_loss": 0.4324, "val_loss": 0.5362, "val_accuracy": 0.0 },
    { "epoch": 2, "train_loss": 0.2608, "val_loss": 1.0352, "val_accuracy": 0.0 },
    { "epoch": 3, "train_loss": 0.1977, "val_loss": 0.7286, "val_accuracy": 0.0 },
    { "epoch": 4, "train_loss": 0.1505, "val_loss": 0.3992, "val_accuracy": 25.0 },
    { "epoch": 5, "train_loss": 0.1331, "val_loss": 0.3993, "val_accuracy": 25.0 },
    { "epoch": 6, "train_loss": 0.1149, "val_loss": 0.3497, "val_accuracy": 25.0 },
    { "epoch": 7, "train_loss": 0.1103, "val_loss": 0.2913, "val_accuracy": 50.0 },
    { "epoch": 8, "train_loss": 0.0893, "val_loss": 0.2356, "val_accuracy": 75.0 },
    { "epoch": 9, "train_loss": 0.0854, "val_loss": 0.1948, "val_accuracy": 100.0 },
    { "epoch": 10, "train_loss": 0.0827, "val_loss": 0.1683, "val_accuracy": 100.0 }
  ]
};

const GAT_DATA = {
  "mean_fingertip_attention": 0.419,
  "mean_palm_attention": 0.2703,
  "attention_ratio": 1.55,
  "sample_edges": [
    { "source": 0, "target": 1, "source_name": "Wrist", "target_name": "Thumb CMC", "attention_weight": 0.3339, "category": "Palm/Wrist" },
    { "source": 1, "target": 0, "source_name": "Thumb CMC", "target_name": "Wrist", "attention_weight": 0.1650, "category": "Palm/Wrist" },
    { "source": 1, "target": 2, "source_name": "Thumb CMC", "target_name": "Thumb MCP", "attention_weight": 0.3509, "category": "Intermediate" },
    { "source": 2, "target": 1, "source_name": "Thumb MCP", "target_name": "Thumb CMC", "attention_weight": 0.3324, "category": "Intermediate" },
    { "source": 2, "target": 3, "source_name": "Thumb MCP", "target_name": "Thumb IP", "attention_weight": 0.4502, "category": "Intermediate" },
    { "source": 3, "target": 2, "source_name": "Thumb IP", "target_name": "Thumb MCP", "attention_weight": 0.2994, "category": "Intermediate" },
    { "source": 3, "target": 4, "source_name": "Thumb IP", "target_name": "Thumb Tip", "attention_weight": 0.5387, "category": "Fingertip" },
    { "source": 4, "target": 3, "source_name": "Thumb Tip", "target_name": "Thumb IP", "attention_weight": 0.2536, "category": "Fingertip" },
    { "source": 0, "target": 5, "source_name": "Wrist", "target_name": "Index MCP", "attention_weight": 0.3314, "category": "Palm/Wrist" },
    { "source": 5, "target": 0, "source_name": "Index MCP", "target_name": "Wrist", "attention_weight": 0.1668, "category": "Palm/Wrist" },
    { "source": 5, "target": 6, "source_name": "Index MCP", "target_name": "Index PIP", "attention_weight": 0.3499, "category": "Intermediate" },
    { "source": 6, "target": 5, "source_name": "Index PIP", "target_name": "Index MCP", "attention_weight": 0.3339, "category": "Intermediate" },
    { "source": 6, "target": 7, "source_name": "Index PIP", "target_name": "Index DIP", "attention_weight": 0.4466, "category": "Intermediate" },
    { "source": 7, "target": 6, "source_name": "Index DIP", "target_name": "Index PIP", "attention_weight": 0.3011, "category": "Intermediate" },
    { "source": 7, "target": 8, "source_name": "Index DIP", "target_name": "Index Tip", "attention_weight": 0.5371, "category": "Fingertip" }
  ],
  "hypothesis_confirmed": true
};

// ── PART 1: Zachary's Karate Club (34 Nodes, PyG Benchmark) ───────────────────
const KARATE_DATA = {
  nodes: [
    {"id": 0,  "club": "Mr. Hi",  "x": 0.4801, "y": 0.646},
    {"id": 1,  "club": "Mr. Hi",  "x": 0.3609, "y": 0.5673},
    {"id": 2,  "club": "Mr. Hi",  "x": 0.3986, "y": 0.4602},
    {"id": 3,  "club": "Mr. Hi",  "x": 0.2198, "y": 0.6128},
    {"id": 4,  "club": "Mr. Hi",  "x": 0.7552, "y": 0.7542},
    {"id": 5,  "club": "Mr. Hi",  "x": 0.6361, "y": 0.878},
    {"id": 6,  "club": "Mr. Hi",  "x": 0.6984, "y": 0.8503},
    {"id": 7,  "club": "Mr. Hi",  "x": 0.2162, "y": 0.5579},
    {"id": 8,  "club": "Mr. Hi",  "x": 0.4156, "y": 0.3817},
    {"id": 9,  "club": "Officer", "x": 0.2137, "y": 0.1898},
    {"id": 10, "club": "Mr. Hi",  "x": 0.8099, "y": 0.8181},
    {"id": 11, "club": "Mr. Hi",  "x": 0.2149, "y": 0.8105},
    {"id": 12, "club": "Mr. Hi",  "x": 0.05,   "y": 0.7196},
    {"id": 13, "club": "Mr. Hi",  "x": 0.334,  "y": 0.4889},
    {"id": 14, "club": "Officer", "x": 0.4478, "y": 0.0712},
    {"id": 15, "club": "Officer", "x": 0.4698, "y": 0.1219},
    {"id": 16, "club": "Mr. Hi",  "x": 0.7791, "y": 0.95},
    {"id": 17, "club": "Mr. Hi",  "x": 0.391,  "y": 0.8164},
    {"id": 18, "club": "Officer", "x": 0.2989, "y": 0.05},
    {"id": 19, "club": "Mr. Hi",  "x": 0.6279, "y": 0.5279},
    {"id": 20, "club": "Officer", "x": 0.1749, "y": 0.084},
    {"id": 21, "club": "Mr. Hi",  "x": 0.3015, "y": 0.7352},
    {"id": 22, "club": "Officer", "x": 0.1218, "y": 0.1573},
    {"id": 23, "club": "Officer", "x": 0.6281, "y": 0.1594},
    {"id": 24, "club": "Officer", "x": 0.95,   "y": 0.2334},
    {"id": 25, "club": "Officer", "x": 0.8329, "y": 0.2041},
    {"id": 26, "club": "Officer", "x": 0.8414, "y": 0.2709},
    {"id": 27, "club": "Officer", "x": 0.6898, "y": 0.2625},
    {"id": 28, "club": "Officer", "x": 0.4111, "y": 0.3205},
    {"id": 29, "club": "Officer", "x": 0.7038, "y": 0.1722},
    {"id": 30, "club": "Officer", "x": 0.2649, "y": 0.3372},
    {"id": 31, "club": "Officer", "x": 0.6561, "y": 0.317},
    {"id": 32, "club": "Officer", "x": 0.4284, "y": 0.1982},
    {"id": 33, "club": "Officer", "x": 0.4521, "y": 0.2389}
  ],
  edges: [
    {s:0,t:1},{s:0,t:2},{s:0,t:3},{s:0,t:4},{s:0,t:5},{s:0,t:6},{s:0,t:7},{s:0,t:8},
    {s:0,t:10},{s:0,t:11},{s:0,t:12},{s:0,t:13},{s:0,t:17},{s:0,t:19},{s:0,t:21},{s:0,t:31},
    {s:1,t:2},{s:1,t:3},{s:1,t:7},{s:1,t:13},{s:1,t:17},{s:1,t:19},{s:1,t:21},{s:1,t:30},
    {s:2,t:3},{s:2,t:7},{s:2,t:8},{s:2,t:9},{s:2,t:13},{s:2,t:27},{s:2,t:28},{s:2,t:32},
    {s:3,t:7},{s:3,t:12},{s:3,t:13},{s:4,t:6},{s:4,t:10},{s:5,t:6},{s:5,t:10},{s:5,t:16},
    {s:6,t:16},{s:8,t:30},{s:8,t:32},{s:8,t:33},{s:9,t:33},{s:13,t:33},{s:14,t:32},{s:14,t:33},
    {s:15,t:32},{s:15,t:33},{s:18,t:32},{s:18,t:33},{s:19,t:33},{s:20,t:32},{s:20,t:33},
    {s:22,t:32},{s:22,t:33},{s:23,t:25},{s:23,t:27},{s:23,t:29},{s:23,t:32},{s:23,t:33},
    {s:24,t:25},{s:24,t:27},{s:24,t:31},{s:25,t:31},{s:26,t:29},{s:26,t:33},{s:27,t:33},
    {s:28,t:31},{s:28,t:33},{s:29,t:32},{s:29,t:33},{s:30,t:32},{s:30,t:33},{s:31,t:32},
    {s:31,t:33},{s:32,t:33}
  ]
};

const COLOR = {
  mrhi:    "#c2410c",
  officer: "#1d4ed8",
  target:  "#b45309",
  hop1:    "#0284c7",
  hop2:    "#64748b",
  dim:     "#cbd5e1"
};

const NUM_NODES = 34;
let currentTargetNode = 0;
let currentFilter = "all";
let currentMatrixMode = "ahat";

let matrixA = [], matrixATilde = [], matrixAHat = [];
let degrees = [], degreesTilde = [];
const adjMap = Array.from({ length: NUM_NODES }, () => []);

function computeGraphMatrices() {
  for (let i = 0; i < NUM_NODES; i++) {
    matrixA[i]      = new Array(NUM_NODES).fill(0);
    matrixATilde[i] = new Array(NUM_NODES).fill(0);
    matrixAHat[i]   = new Array(NUM_NODES).fill(0);
  }

  KARATE_DATA.edges.forEach(e => {
    matrixA[e.s][e.t] = 1;
    matrixA[e.t][e.s] = 1;
    adjMap[e.s].push(e.t);
    adjMap[e.t].push(e.s);
  });

  for (let i = 0; i < NUM_NODES; i++) {
    degrees[i] = adjMap[i].length;
    for (let j = 0; j < NUM_NODES; j++) {
      matrixATilde[i][j] = matrixA[i][j] + (i === j ? 1 : 0);
    }
    degreesTilde[i] = degrees[i] + 1;
  }

  for (let i = 0; i < NUM_NODES; i++) {
    for (let j = 0; j < NUM_NODES; j++) {
      matrixAHat[i][j] = matrixATilde[i][j] !== 0
        ? matrixATilde[i][j] / Math.sqrt(degreesTilde[i] * degreesTilde[j])
        : 0;
    }
  }

  const maxDeg  = Math.max(...degrees);
  const minDeg  = Math.min(...degrees);
  const avgDeg  = (degrees.reduce((a, b) => a + b, 0) / NUM_NODES).toFixed(2);
  const edgeCount = KARATE_DATA.edges.length;
  const sparsity  = (100 * (1 - (edgeCount * 2) / (NUM_NODES * NUM_NODES))).toFixed(1);

  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("val-num-nodes", NUM_NODES);
  set("val-num-edges", edgeCount);
  set("val-max-deg",   maxDeg);
  set("val-min-deg",   minDeg);
  set("val-avg-deg",   avgDeg);
  set("val-sparsity",  `${sparsity}%`);
}

function svgEl(tag) {
  return document.createElementNS("http://www.w3.org/2000/svg", tag);
}

function getNodeColor(nodeId) {
  const node = KARATE_DATA.nodes[nodeId];
  if (nodeId === currentTargetNode) return COLOR.target;
  const neighbors1 = adjMap[currentTargetNode];
  if (neighbors1.includes(nodeId)) {
    if (currentFilter === "target") return COLOR.dim;
    return COLOR.hop1;
  }
  const neighbors2 = get2HopNeighborhood(currentTargetNode);
  if (neighbors2.includes(nodeId)) {
    if (currentFilter === "target" || currentFilter === "1hop") return COLOR.dim;
    return COLOR.hop2;
  }
  if (currentFilter === "all") {
    return node.club === "Mr. Hi" ? COLOR.mrhi : COLOR.officer;
  }
  return COLOR.dim;
}

function get2HopNeighborhood(target) {
  const hop1 = new Set(adjMap[target]);
  const hop2 = new Set();
  hop1.forEach(n => {
    adjMap[n].forEach(n2 => {
      if (n2 !== target && !hop1.has(n2)) hop2.add(n2);
    });
  });
  return Array.from(hop2);
}

function renderGraphSvg() {
  const svg = document.getElementById("graph-svg");
  if (!svg) return;
  svg.innerHTML = "";

  const W = 640, H = 420;
  const pad = 36;
  const scaleX = x => pad + x * (W - 2 * pad);
  const scaleY = y => pad + y * (H - 2 * pad);

  const edgeGroup = svgEl("g");
  edgeGroup.setAttribute("id", "svg-edges");
  KARATE_DATA.edges.forEach(e => {
    const u = KARATE_DATA.nodes[e.s];
    const v = KARATE_DATA.nodes[e.t];
    const line = svgEl("line");
    line.setAttribute("x1", scaleX(u.x)); line.setAttribute("y1", scaleY(u.y));
    line.setAttribute("x2", scaleX(v.x)); line.setAttribute("y2", scaleY(v.y));
    line.setAttribute("stroke", "#e2e8f0");
    line.setAttribute("stroke-width", "1.5");
    line.setAttribute("id", `edge-${Math.min(e.s,e.t)}-${Math.max(e.s,e.t)}`);
    edgeGroup.appendChild(line);
  });
  svg.appendChild(edgeGroup);

  const particleGroup = svgEl("g");
  particleGroup.setAttribute("id", "svg-particles");
  svg.appendChild(particleGroup);

  const nodeGroup = svgEl("g");
  nodeGroup.setAttribute("id", "svg-nodes");
  KARATE_DATA.nodes.forEach(n => {
    const cx = scaleX(n.x);
    const cy = scaleY(n.y);
    const g = svgEl("g");
    g.setAttribute("class", "node-group");
    g.setAttribute("tabindex", "0");
    g.setAttribute("role", "button");
    g.setAttribute("aria-label", `Node ${n.id}, Faction: ${n.club}, Degree: ${degrees[n.id] || 0}`);
    g.style.cursor = "pointer";

    const circle = svgEl("circle");
    circle.setAttribute("cx", cx); circle.setAttribute("cy", cy);
    circle.setAttribute("r", n.id === currentTargetNode ? "15" : "11");
    circle.setAttribute("fill", getNodeColor(n.id));
    circle.setAttribute("stroke", n.id === currentTargetNode ? "#78350f" : "#ffffff");
    circle.setAttribute("stroke-width", n.id === currentTargetNode ? "3" : "1.5");
    circle.setAttribute("id", `node-circle-${n.id}`);

    const text = svgEl("text");
    text.setAttribute("x", cx); text.setAttribute("y", cy + 4);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", n.id === currentTargetNode ? "11" : "9");
    text.setAttribute("font-weight", "600");
    text.setAttribute("font-family", "var(--font-mono)");
    text.setAttribute("fill", "#ffffff");
    text.setAttribute("id", `node-text-${n.id}`);
    text.textContent = n.id;

    g.appendChild(circle);
    g.appendChild(text);

    g.addEventListener("click", () => selectNode(n.id));
    g.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter" || evt.key === " ") {
        evt.preventDefault();
        selectNode(n.id);
      }
    });

    nodeGroup.appendChild(g);
  });
  svg.appendChild(nodeGroup);
}

function updateGraphColors() {
  KARATE_DATA.nodes.forEach(n => {
    const circle = document.getElementById(`node-circle-${n.id}`);
    if (circle) {
      circle.setAttribute("fill", getNodeColor(n.id));
      circle.setAttribute("r", n.id === currentTargetNode ? "15" : "11");
      circle.setAttribute("stroke", n.id === currentTargetNode ? "#78350f" : "#ffffff");
      circle.setAttribute("stroke-width", n.id === currentTargetNode ? "3" : "1.5");
    }
  });

  const hop1 = adjMap[currentTargetNode];
  KARATE_DATA.edges.forEach(e => {
    const id = `edge-${Math.min(e.s,e.t)}-${Math.max(e.s,e.t)}`;
    const line = document.getElementById(id);
    if (!line) return;
    const isIncident = (e.s === currentTargetNode && hop1.includes(e.t)) ||
                       (e.t === currentTargetNode && hop1.includes(e.s));
    if (isIncident) {
      line.setAttribute("stroke", "#b45309");
      line.setAttribute("stroke-width", "2.5");
    } else {
      line.setAttribute("stroke", "#e2e8f0");
      line.setAttribute("stroke-width", "1.5");
    }
  });
}

function selectNode(id) {
  currentTargetNode = id;
  updateGraphColors();
  updateInspector(id);
  renderSubmatrixTable();
  renderMatrixHeatmap();
}

function filterNeighborhood(mode) {
  currentFilter = mode;
  ["btn-show-all", "btn-show-target", "btn-show-1hop", "btn-show-2hop"].forEach(btnId => {
    const el = document.getElementById(btnId);
    if (el) el.classList.remove("active");
  });
  const activeMap = {
    all: "btn-show-all",
    target: "btn-show-target",
    "1hop": "btn-show-1hop",
    "2hop": "btn-show-2hop"
  };
  const activeEl = document.getElementById(activeMap[mode]);
  if (activeEl) activeEl.classList.add("active");
  updateGraphColors();
}

function updateInspector(nodeId) {
  const node = KARATE_DATA.nodes[nodeId];
  const hop1 = adjMap[nodeId];
  const hop2 = get2HopNeighborhood(nodeId);

  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set("insp-id", nodeId);
  set("insp-club", node.club);
  set("insp-deg", hop1.length);
  set("insp-neighbors", hop1.join(", "));
  set("insp-2hop", hop2.length > 0 ? hop2.join(", ") : "None");

  const formulaEl = document.getElementById("insp-mpnn-formula");
  if (formulaEl) {
    delete formulaEl.dataset.latexProcessed;
    delete formulaEl.dataset.inspectorAttached;
    formulaEl.innerHTML = `$$\\bar{m}_{${nodeId}} = \\frac{1}{${hop1.length}} \\sum_{u \\in \\mathcal{N}(${nodeId})} h_u$$`;
    renderMath(formulaEl);
  }
}

function simulateMessagePassing() {
  const pGroup = document.getElementById("svg-particles");
  if (!pGroup) return;
  pGroup.innerHTML = "";

  const W = 640, H = 420;
  const pad = 36;
  const scaleX = x => pad + x * (W - 2 * pad);
  const scaleY = y => pad + y * (H - 2 * pad);

  const target = KARATE_DATA.nodes[currentTargetNode];
  const tx = scaleX(target.x);
  const ty = scaleY(target.y);

  const hop1 = adjMap[currentTargetNode];
  const duration = 600;

  hop1.forEach((neighborId, i) => {
    const neighbor = KARATE_DATA.nodes[neighborId];
    const sx = scaleX(neighbor.x);
    const sy = scaleY(neighbor.y);

    const particle = svgEl("circle");
    particle.setAttribute("r", "5");
    particle.setAttribute("fill", "#b45309");
    particle.setAttribute("cx", sx);
    particle.setAttribute("cy", sy);
    pGroup.appendChild(particle);

    const startDelay = i * 40;
    const startTime = performance.now() + startDelay;

    function step(now) {
      if (now < startTime) {
        requestAnimationFrame(step);
        return;
      }
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

      particle.setAttribute("cx", sx + (tx - sx) * ease);
      particle.setAttribute("cy", sy + (ty - sy) * ease);

      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        particle.remove();
        if (i === hop1.length - 1) {
          flashTargetNode();
        }
      }
    }
    requestAnimationFrame(step);
  });
}

function flashTargetNode() {
  const circle = document.getElementById(`node-circle-${currentTargetNode}`);
  if (!circle) return;
  const origR = circle.getAttribute("r");
  circle.setAttribute("r", "20");
  circle.setAttribute("fill", "#f59e0b");
  setTimeout(() => {
    circle.setAttribute("r", origR);
    circle.setAttribute("fill", getNodeColor(currentTargetNode));
  }, 250);
}

// ── PART 2: Scrollbar-Free Matrix Heatmap & Live Derivation ───────────────────
let matrixCanvas, matrixCtx;
let hoveredCell = { u: 0, v: 1 };

function initMatrixHeatmap() {
  matrixCanvas = document.getElementById("matrix-canvas");
  if (!matrixCanvas) return;
  matrixCtx = matrixCanvas.getContext("2d");

  matrixCanvas.addEventListener("mousemove", (e) => {
    const rect = matrixCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cellSize = 340 / NUM_NODES; // 10px per cell
    const u = Math.floor(y / cellSize);
    const v = Math.floor(x / cellSize);

    if (u >= 0 && u < NUM_NODES && v >= 0 && v < NUM_NODES) {
      hoveredCell = { u, v };
      const val = getMatrixValue(u, v, currentMatrixMode);
      inspectMatrixCell(u, v, val, currentMatrixMode);
      renderMatrixHeatmap();
    }
  });

  matrixCanvas.addEventListener("click", () => {
    selectNode(hoveredCell.u);
  });

  renderMatrixHeatmap();
  renderSubmatrixTable();
}

function getMatrixValue(i, j, mode) {
  if (mode === "a") return matrixA[i][j];
  if (mode === "atilde") return matrixATilde[i][j];
  return matrixAHat[i][j];
}

function renderMatrixHeatmap() {
  if (!matrixCtx) return;
  const W = 340, H = 340;
  const cellSize = W / NUM_NODES; // 10px

  matrixCtx.clearRect(0, 0, W, H);

  // Render all 34x34 cells
  for (let i = 0; i < NUM_NODES; i++) {
    for (let j = 0; j < NUM_NODES; j++) {
      const val = getMatrixValue(i, j, currentMatrixMode);
      const x = j * cellSize;
      const y = i * cellSize;

      if (val === 0) {
        matrixCtx.fillStyle = "#ffffff";
      } else {
        if (currentMatrixMode === "ahat") {
          // Normalize intensity: max value around 0.35
          const intensity = Math.min(val / 0.3, 1.0);
          const r = Math.round(254 - intensity * (254 - 180));
          const g = Math.round(243 - intensity * (243 - 83));
          const b = Math.round(199 - intensity * (199 - 9));
          matrixCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        } else {
          matrixCtx.fillStyle = i === j ? "#fde68a" : "#fed7aa";
        }
      }

      matrixCtx.fillRect(x, y, cellSize, cellSize);

      // Subtle cell boundary
      matrixCtx.strokeStyle = "#e2e8f0";
      matrixCtx.lineWidth = 0.5;
      matrixCtx.strokeRect(x, y, cellSize, cellSize);
    }
  }

  // Crosshair highlight for target node row and hovered cell
  const targetY = currentTargetNode * cellSize;
  matrixCtx.strokeStyle = "#b45309";
  matrixCtx.lineWidth = 1.5;
  matrixCtx.strokeRect(0, targetY, W, cellSize);

  // Hovered cell reticle
  if (hoveredCell.u >= 0 && hoveredCell.v >= 0) {
    const hx = hoveredCell.v * cellSize;
    const hy = hoveredCell.u * cellSize;
    matrixCtx.strokeStyle = "#0284c7";
    matrixCtx.lineWidth = 2;
    matrixCtx.strokeRect(hx, hy, cellSize, cellSize);
  }
}

function showMatrix(mode) {
  currentMatrixMode = mode;
  ["btn-mat-ahat", "btn-mat-atilde", "btn-mat-a"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active");
  });
  const activeMap = { ahat: "btn-mat-ahat", atilde: "btn-mat-atilde", a: "btn-mat-a" };
  const el = document.getElementById(activeMap[mode]);
  if (el) el.classList.add("active");

  renderMatrixHeatmap();
  renderSubmatrixTable();
  inspectMatrixCell(hoveredCell.u, hoveredCell.v, getMatrixValue(hoveredCell.u, hoveredCell.v, mode), mode);
}

function inspectMatrixCell(u, v, val, mode) {
  const title = document.getElementById("insp-cell-title");
  const formula = document.getElementById("insp-cell-formula");
  if (!title || !formula) return;

  const duTilde = degreesTilde[u];
  const dvTilde = degreesTilde[v];

  if (mode === "a") {
    title.textContent = `Binary Adjacency A[${u}, ${v}] = ${val}`;
    formula.innerHTML = `$$A_{${u},${v}} = ${val} \\quad (${u} \\text{ and } ${v} ${val === 1 ? "\\text{are connected}" : "\\text{not connected}"})$$`;
  } else if (mode === "atilde") {
    title.textContent = `Self-Loop Adjacency A~[${u}, ${v}] = ${val}`;
    formula.innerHTML = `$$\\tilde{A}_{${u},${v}} = A_{${u},${v}} + I_{${u},${v}} = ${val}$$`;
  } else {
    title.textContent = `Symmetric Normalized A^[${u}, ${v}] = ${val.toFixed(4)}`;
    if (val === 0) {
      formula.innerHTML = `$$\\hat{A}_{${u},${v}} = 0 \\quad (\\tilde{A}_{${u},${v}} = 0, \\text{ no connection})$$`;
    } else {
      formula.innerHTML = `$$\\hat{A}_{${u},${v}} = \\frac{\\tilde{A}_{${u},${v}}}{\\sqrt{\\tilde{D}_{${u},${u}} \\tilde{D}_{${v},${v}}}} = \\frac{1}{\\sqrt{${duTilde} \\times ${dvTilde}}} = \\frac{1}{${Math.sqrt(duTilde * dvTilde).toFixed(3)}} = ${val.toFixed(4)}$$`;
    }
  }
  delete formula.dataset.latexProcessed;
  delete formula.dataset.inspectorAttached;
  renderMath(formula);
}

// ── Target Node Submatrix Table (Scrollbar-Free) ──────────────────────────────
function renderSubmatrixTable() {
  const tbody = document.getElementById("submatrix-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const u = currentTargetNode;
  const neighbors = [u, ...adjMap[u]]; // self + 1-hop neighbors

  neighbors.forEach(v => {
    const tr = document.createElement("tr");
    const aVal = matrixA[u][v];
    const atildeVal = matrixATilde[u][v];
    const ahatVal = matrixAHat[u][v];

    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td><strong>v${v}</strong> ${v === u ? "(Self)" : ""}</td>
      <td>${KARATE_DATA.nodes[v].club}</td>
      <td>${degrees[v]}</td>
      <td>${aVal}</td>
      <td>${atildeVal}</td>
      <td style="font-weight:700;color:var(--accent-primary);">${ahatVal.toFixed(4)}</td>
    `;
    tr.addEventListener("mouseenter", () => {
      hoveredCell = { u, v };
      renderMatrixHeatmap();
      inspectMatrixCell(u, v, getMatrixValue(u, v, currentMatrixMode), currentMatrixMode);
    });
    tbody.appendChild(tr);
  });
}

// ── PART 3: Pure-SVG Line Chart Renderer ──────────────────────────────────────
function renderLineChart(svgId, epochs, series) {
  const svg = document.getElementById(svgId);
  if (!svg) return;

  const W = svg.parentElement.clientWidth - 40 || 600;
  const H = 180;
  const padL = 48, padR = 16, padT = 12, padB = 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";

  const allVals = series.flatMap(s => epochs.map(e => e[s.key])).filter(v => v !== undefined);
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const range  = maxVal - minVal || 1;

  const xScale = i => padL + (i / (epochs.length - 1)) * (W - padL - padR);
  const yScale = v => padT + ((maxVal - v) / range) * (H - padT - padB);

  // 5 Horizontal Grid Lines
  for (let k = 0; k <= 4; k++) {
    const y = padT + k * (H - padT - padB) / 4;
    const val = maxVal - k * range / 4;
    const line = svgEl("line");
    line.setAttribute("x1", padL); line.setAttribute("x2", W - padR);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("stroke", "#e2e8f0"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);

    const label = svgEl("text");
    label.setAttribute("x", padL - 6); label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", "10"); label.setAttribute("fill", "#94a3b8");
    label.setAttribute("font-family", "var(--font-mono)");
    label.textContent = val.toFixed(1);
    svg.appendChild(label);
  }

  // Epoch numbers on X-axis
  epochs.forEach((e, i) => {
    const x = xScale(i);
    const label = svgEl("text");
    label.setAttribute("x", x); label.setAttribute("y", H - padB + 16);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "10"); label.setAttribute("fill", "#94a3b8");
    label.setAttribute("font-family", "var(--font-mono)");
    label.textContent = e.epoch;
    svg.appendChild(label);
  });

  // Series Polylines & Dots
  series.forEach(s => {
    const points = epochs.map((e, i) => `${xScale(i)},${yScale(e[s.key])}`).join(" ");
    const polyline = svgEl("polyline");
    polyline.setAttribute("points", points);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", s.color);
    polyline.setAttribute("stroke-width", "2");
    if (s.dashed) polyline.setAttribute("stroke-dasharray", "5 3");
    svg.appendChild(polyline);

    epochs.forEach((e, i) => {
      const dot = svgEl("circle");
      dot.setAttribute("cx", xScale(i)); dot.setAttribute("cy", yScale(e[s.key]));
      dot.setAttribute("r", "3"); dot.setAttribute("fill", s.color);
      dot.setAttribute("stroke", "#fff"); dot.setAttribute("stroke-width", "1.5");
      svg.appendChild(dot);
    });
  });
}

// ── PART 4: Populate Real Results & Charts ────────────────────────────────────
function renderResults() {
  // GCN Training Results
  const gcnLoading = document.getElementById("gcn-chart-loading");
  if (gcnLoading) gcnLoading.classList.remove("active");
  const gcnWrap = document.getElementById("gcn-chart-wrap");
  if (gcnWrap) gcnWrap.style.display = "block";

  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("gcn-final-acc", `${STATIC_GCN_DATA.final_test_accuracy}%`);
  set("gcn-train-samples", STATIC_GCN_DATA.train_samples);
  set("gcn-test-samples", STATIC_GCN_DATA.test_samples);
  set("gcn-epochs", STATIC_GCN_DATA.epochs);
  set("gcn-duration", `${STATIC_GCN_DATA.duration_seconds}s`);
  set("gcn-device-tag", `${STATIC_GCN_DATA.device} · ${STATIC_GCN_DATA.device_name}`);

  renderLineChart("gcn-chart-svg", STATIC_GCN_DATA.history, [
    { key: "train_loss", color: "#b45309", label: "Train Loss" },
    { key: "train_acc",  color: "#15803d", label: "Train Acc" },
    { key: "test_acc",   color: "#0284c7", dashed: true, label: "Test Acc" }
  ]);

  // ST-GCN Results
  const stgcnLoading = document.getElementById("stgcn-chart-loading");
  if (stgcnLoading) stgcnLoading.classList.remove("active");
  const stgcnWrap = document.getElementById("stgcn-chart-wrap");
  if (stgcnWrap) stgcnWrap.style.display = "block";

  set("stgcn-final-acc", `${STGCN_DATA.final_val_accuracy}%`);
  set("stgcn-final-loss", STGCN_DATA.final_val_loss);
  set("stgcn-classes", STGCN_DATA.num_gloss_classes);
  set("stgcn-epochs", STGCN_DATA.epochs);
  set("stgcn-duration", `${STGCN_DATA.duration_seconds}s`);
  set("stgcn-device-tag", `${STGCN_DATA.device} · ${STGCN_DATA.device_name}`);

  const tagsWrap = document.getElementById("stgcn-gloss-tags");
  const glossWrap = document.getElementById("stgcn-glosses-wrap");
  if (tagsWrap && glossWrap) {
    tagsWrap.innerHTML = "";
    STGCN_DATA.glosses.forEach(g => {
      const span = document.createElement("span");
      span.className = "badge";
      span.style.fontFamily = "var(--font-mono)";
      span.textContent = g;
      tagsWrap.appendChild(span);
    });
    glossWrap.style.display = "block";
  }

  renderLineChart("stgcn-chart-svg", STGCN_DATA.history, [
    { key: "train_loss",   color: "#b45309", label: "Train Loss" },
    { key: "val_loss",     color: "#7c3aed", label: "Val Loss" },
    { key: "val_accuracy", color: "#15803d", label: "Val Acc" }
  ]);

  // GAT Attention Weights Table
  const gatLoading = document.getElementById("gat-attention-loading");
  if (gatLoading) gatLoading.classList.remove("active");
  const gatWrap = document.getElementById("gat-attention-wrap");
  if (gatWrap) gatWrap.style.display = "block";

  set("gat-fingertip-attn", GAT_DATA.mean_fingertip_attention.toFixed(3));
  set("gat-palm-attn", GAT_DATA.mean_palm_attention.toFixed(3));
  set("gat-ratio", `${GAT_DATA.attention_ratio.toFixed(2)}x`);
  set("gat-confirmed", GAT_DATA.hypothesis_confirmed ? "Yes" : "No");

  const tbody = document.getElementById("gat-attention-tbody");
  if (tbody) {
    tbody.innerHTML = "";
    const maxAttn = Math.max(...GAT_DATA.sample_edges.map(e => e.attention_weight));
    const catColor = { "Fingertip": "#15803d", "Intermediate": "#0284c7", "Palm/Wrist": "#b45309" };

    GAT_DATA.sample_edges.forEach(e => {
      const tr = document.createElement("tr");
      const barWidth = Math.round((e.attention_weight / maxAttn) * 90);
      const color = catColor[e.category] || "#475569";
      tr.innerHTML = `
        <td>${e.source_name}</td>
        <td>${e.target_name}</td>
        <td><span style="font-size:0.75rem;font-weight:600;color:${color};">${e.category}</span></td>
        <td style="font-family:var(--font-mono);font-weight:600;">${e.attention_weight.toFixed(4)}</td>
        <td><span class="attn-bar" style="width:${barWidth}px;background:${color};"></span></td>
      `;
      tbody.appendChild(tr);
    });
  }
}

// ── PART 5: KaTeX & Scroll-Spy Navigation ─────────────────────────────────────
function renderMath(target = document.body) {
  if (window.renderAllMath) {
    window.renderAllMath(target);
  } else if (window.renderMathInElement) {
    try {
      window.renderMathInElement(target, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$",  right: "$",  display: false },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    } catch (err) {
      console.warn("KaTeX:", err);
    }
  } else {
    setTimeout(() => renderMath(target), 100);
  }
}

function initScrollSpy() {
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll("#sidebar a");

  window.addEventListener("scroll", () => {
    let currentId = "";
    const scrollPos = window.scrollY + 120;

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute("id");
      }
    });

    if (currentId) {
      navLinks.forEach(link => {
        if (link.getAttribute("href") === `#${currentId}`) {
          link.classList.add("active");
        } else {
          link.classList.remove("active");
        }
      });
    }
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  computeGraphMatrices();
  renderGraphSvg();
  selectNode(0);
  initMatrixHeatmap();
  inspectMatrixCell(0, 1, matrixAHat[0][1], "ahat");
  renderResults();
  initScrollSpy();
  setTimeout(() => renderMath(), 100);
});
