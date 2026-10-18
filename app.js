/**
 * app.js — FDTD Educational Web App
 *
 * Orchestrates:
 *  • Three.js 3D scene (orthogonal slice planes on a Yee grid)
 *  • Yee-cell interactive diagram (separate Three.js scene)
 *  • Hero animated background
 *  • WebWorker communication with fdtd_worker.js
 *  • All UI controls, experiments, FFT analysis, algorithm walkthrough
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ══════════════════════════════════════════════════════════════════════════════
//  CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════

const C0 = 299792458; // m/s

const GRID_CONFIGS = {
  small:  { Nx: 60,  Ny: 60,  Nz: 60,  dx: 3e-3, label: '60³' },
  medium: { Nx: 80,  Ny: 80,  Nz: 80,  dx: 3e-3, label: '80³' },
  large:  { Nx: 100, Ny: 100, Nz: 100, dx: 3e-3, label: '100³' }
};

// ══════════════════════════════════════════════════════════════════════════════
//  APPLICATION STATE
// ══════════════════════════════════════════════════════════════════════════════

const state = {
  gridKey:         'small',
  gridConfig:      { ...GRID_CONFIGS.small },
  running:         false,
  stepsPerFrame:   3,
  sliceIdx:        { x: 30, y: 30, z: 30 },
  maxE:            1e-12,
  probeHistory:    [],
  tdHistory:       [],
  workerReady:     false,
  activeMaterial:  'free_space',
};

// ══════════════════════════════════════════════════════════════════════════════
//  WEB WORKER
// ══════════════════════════════════════════════════════════════════════════════

let worker = null;

function initWorker() {
  if (worker) worker.terminate();
  worker = new Worker('fdtd_worker.js');
  worker.onmessage = handleWorkerMessage;

  const cfg = state.gridConfig;
  const sliceIdx = {
    sliceX: Math.floor(cfg.Nx / 2),
    sliceY: Math.floor(cfg.Ny / 2),
    sliceZ: Math.floor(cfg.Nz / 2),
  };
  state.sliceIdx = { x: sliceIdx.sliceX, y: sliceIdx.sliceY, z: sliceIdx.sliceZ };

  worker.postMessage({ type: 'init', data: { ...cfg, ...sliceIdx } });

  // Update slider max values
  ['x', 'y', 'z'].forEach(axis => {
    const el = document.getElementById(`slice-${axis}`);
    if (el) {
      el.max   = cfg[`N${axis.toUpperCase()}`] - 1;
      el.value = state.sliceIdx[axis];
    }
  });
}

function handleWorkerMessage({ data: msg }) {
  switch (msg.type) {
    case 'ready':    onWorkerReady(msg.data); break;
    case 'frame':    onFrame(msg.data);       break;
    case 'reset_done': onResetDone();         break;
    default: break;
  }
}

function onWorkerReady(info) {
  state.workerReady = true;

  const el = id => document.getElementById(id);
  if (el('info-grid'))    el('info-grid').textContent = state.gridConfig.label;
  if (el('info-dx'))      el('info-dx').textContent   = `${(info.dx * 1e3).toFixed(1)} mm`;
  if (el('info-dt'))      el('info-dt').textContent   = `${(info.dt * 1e12).toFixed(2)} ps`;
  if (el('courant-number')) el('courant-number').textContent = `S = ${info.Courant.toFixed(4)}`;
  if (el('courant-display')) el('courant-display').textContent = `S = ${info.Courant.toFixed(4)}`;

  // Hide loading overlay
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.classList.add('hidden');

  // Apply default material (free space — no-op), but reset fields
  worker.postMessage({ type: 'reset' });
}

function onFrame(data) {
  const { xySlice, xzSlice, yzSlice, maxE, timeStep, time, probeEz } = data;

  // Smooth running max for colourmap normalization
  state.maxE = state.maxE * 0.92 + maxE * 0.08;
  const normMax = Math.max(state.maxE, 1e-12);

  // Push to transferred arrays directly into DataTextures
  updateSliceTex('xy', xySlice, normMax);
  updateSliceTex('xz', xzSlice, normMax);
  updateSliceTex('yz', yzSlice, normMax);

  // HUD updates
  setHud('time-display', `t = ${(time * 1e9).toFixed(2)} ns`);
  setHud('step-display', `Step: ${timeStep}`);
  setHud('info-maxe',    `${maxE.toExponential(2)}`);

  // Probe history
  state.probeHistory.push(probeEz);
  state.tdHistory.push(probeEz);
  if (state.probeHistory.length > 256) state.probeHistory.shift();

  drawProbeChart();

  // Continue loop
  if (state.running) scheduleNextStep();
}

function onResetDone() {
  state.maxE = 1e-12;
  state.probeHistory = [];
  state.tdHistory    = [];
  drawProbeChart();
  clearTDChart();
}

function scheduleNextStep() {
  worker.postMessage({ type: 'step', data: { stepsPerFrame: state.stepsPerFrame } });
}

function setHud(elemId, text) {
  const el = document.getElementById(elemId);
  if (el) el.textContent = text;
}

// ══════════════════════════════════════════════════════════════════════════════
//  THREE.JS — MAIN SIMULATOR SCENE
// ══════════════════════════════════════════════════════════════════════════════

let simRenderer, simScene, simCamera, simControls;
const sliceMeshes   = {};   // name → THREE.Mesh
const sliceTextures = {};   // name → { tex: DataTexture, data: Float32Array }
const sliceUniforms = {};   // name → uniform object
let boxMesh = null;
let axesHelper = null;

// GLSL shaders for field slice planes
const SLICE_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const SLICE_FRAG = `
uniform sampler2D fieldTex;
uniform float maxVal;
varying vec2 vUv;

vec3 colormap(float v) {
  float t = clamp(v, -1.0, 1.0);
  float a = pow(abs(t), 0.55);
  if (t > 0.0) {
    // Positive: black → cyan
    return vec3(0.0, 0.83 * a, 1.0 * a);
  } else {
    // Negative: black → magenta
    return vec3(1.0 * a, 0.1 * a, 0.5 * a);
  }
}

void main() {
  float raw = texture2D(fieldTex, vUv).r;
  float v   = raw / maxVal;
  vec3  col = colormap(v);
  float alpha = 0.12 + 0.88 * pow(abs(clamp(v, -1.0, 1.0)), 0.45);
  gl_FragColor = vec4(col, alpha);
}`;

function initSimScene() {
  const container = document.getElementById('sim-canvas-container');
  if (!container) return;

  const W = container.clientWidth  || 800;
  const H = container.clientHeight || 600;

  // Renderer
  simRenderer = new THREE.WebGLRenderer({ antialias: true });
  simRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  simRenderer.setSize(W, H);
  simRenderer.setClearColor(0x020408, 1);
  container.appendChild(simRenderer.domElement);

  // Scene
  simScene = new THREE.Scene();

  // Camera
  simCamera = new THREE.PerspectiveCamera(50, W / H, 0.01, 50);
  simCamera.position.set(1.9, 1.4, 1.9);
  simCamera.lookAt(0, 0, 0);

  // Controls
  simControls = new OrbitControls(simCamera, simRenderer.domElement);
  simControls.enableDamping    = true;
  simControls.dampingFactor    = 0.07;
  simControls.minDistance      = 0.5;
  simControls.maxDistance      = 6;
  simControls.enablePan        = true;
  simControls.panSpeed         = 0.4;

  // Bounding box
  rebuildBoundingBox();

  // Axes
  axesHelper = new THREE.AxesHelper(0.7);
  simScene.add(axesHelper);

  // Slice planes
  rebuildSlicePlanes();

  // Subtle grid on floor
  addFloorGrid();

  // Lighting (minimal — ShaderMaterial doesn't use it but adds depth to other objects)
  simScene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0x00d4ff, 0.4);
  dir.position.set(2, 4, 2);
  simScene.add(dir);

  // Resize handler
  window.addEventListener('resize', onSimResize);

  // Render loop
  renderSim();
}

function rebuildBoundingBox() {
  if (boxMesh) simScene.remove(boxMesh);
  const geom  = new THREE.BoxGeometry(1, 1, 1);
  const edges  = new THREE.EdgesGeometry(geom);
  const mat    = new THREE.LineBasicMaterial({ color: 0x00d4ff, opacity: 0.2, transparent: true });
  boxMesh = new THREE.LineSegments(edges, mat);
  simScene.add(boxMesh);
}

function addFloorGrid() {
  const pts = [];
  const N   = 10;
  for (let i = 0; i <= N; i++) {
    const t = i / N - 0.5;
    pts.push(new THREE.Vector3(t, -0.5, -0.5), new THREE.Vector3(t, -0.5,  0.5));
    pts.push(new THREE.Vector3(-0.5, -0.5, t),  new THREE.Vector3( 0.5, -0.5, t));
  }
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const mat  = new THREE.LineBasicMaterial({ color: 0x111a30, opacity: 0.5, transparent: true });
  simScene.add(new THREE.LineSegments(geom, mat));
}

function rebuildSlicePlanes() {
  // Remove old
  ['xy', 'xz', 'yz'].forEach(name => {
    if (sliceMeshes[name]) simScene.remove(sliceMeshes[name]);
  });

  createSlicePlane('xy');
  createSlicePlane('xz');
  createSlicePlane('yz');
}

function createSlicePlane(name) {
  const { Nx, Ny, Nz } = state.gridConfig;
  let texW, texH;
  if      (name === 'xy') { texW = Nx; texH = Ny; }
  else if (name === 'xz') { texW = Nx; texH = Nz; }
  else                    { texW = Ny; texH = Nz; }

  const data    = new Float32Array(texW * texH);
  const tex     = new THREE.DataTexture(data, texW, texH, THREE.RedFormat, THREE.FloatType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;

  const uniforms = {
    fieldTex: { value: tex },
    maxVal:   { value: 1.0 },
  };

  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader:   SLICE_VERT,
    fragmentShader: SLICE_FRAG,
    transparent:    true,
    side:           THREE.DoubleSide,
    depthWrite:     false,
  });

  const geom = new THREE.PlaneGeometry(1, 1);
  const mesh = new THREE.Mesh(geom, mat);

  // Orient & position each plane at the slice index position
  const { x, y, z } = state.sliceIdx;
  if (name === 'xy') {
    // Default XY plane, move along z
    mesh.position.z = z / Nz - 0.5;
  } else if (name === 'xz') {
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = y / Ny - 0.5;
  } else {
    // YZ
    mesh.rotation.y = Math.PI / 2;
    mesh.position.x = x / Nx - 0.5;
  }

  simScene.add(mesh);
  sliceMeshes[name]   = mesh;
  sliceTextures[name] = { tex, data, texW, texH };
  sliceUniforms[name] = uniforms;
}

function updateSliceTex(name, newData, maxVal) {
  const st = sliceTextures[name];
  if (!st || !newData) return;
  st.data.set(newData);
  st.tex.needsUpdate = true;
  sliceUniforms[name].maxVal.value = maxVal;
}

function renderSim() {
  requestAnimationFrame(renderSim);
  simControls.update();
  simRenderer.render(simScene, simCamera);
}

function onSimResize() {
  const container = document.getElementById('sim-canvas-container');
  if (!container || !simRenderer) return;
  const W = container.clientWidth;
  const H = container.clientHeight;
  simCamera.aspect = W / H;
  simCamera.updateProjectionMatrix();
  simRenderer.setSize(W, H);
}

// ══════════════════════════════════════════════════════════════════════════════
//  THREE.JS — YEE CELL DIAGRAM
// ══════════════════════════════════════════════════════════════════════════════

let yeeCellRenderer, yeeCellScene, yeeCellCamera, yeeCellControls;

function initYeeCellScene() {
  const container = document.getElementById('yee-cell-canvas');
  if (!container) return;

  const W = container.clientWidth  || 320;
  const H = container.clientHeight || 300;

  yeeCellRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  yeeCellRenderer.setPixelRatio(window.devicePixelRatio);
  yeeCellRenderer.setSize(W, H);
  yeeCellRenderer.setClearColor(0x000000, 0);
  container.appendChild(yeeCellRenderer.domElement);

  yeeCellScene    = new THREE.Scene();
  yeeCellCamera   = new THREE.PerspectiveCamera(45, W / H, 0.01, 100);
  yeeCellCamera.position.set(2.5, 2.0, 2.5);
  yeeCellCamera.lookAt(0, 0, 0);

  yeeCellControls = new OrbitControls(yeeCellCamera, yeeCellRenderer.domElement);
  yeeCellControls.enableDamping   = true;
  yeeCellControls.dampingFactor   = 0.06;
  yeeCellControls.autoRotate      = true;
  yeeCellControls.autoRotateSpeed = 1.8;
  yeeCellControls.enableZoom      = true;

  // Lighting
  yeeCellScene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const pt = new THREE.PointLight(0x00d4ff, 1, 20);
  pt.position.set(3, 3, 3);
  yeeCellScene.add(pt);

  buildYeeCellGeometry();

  function animYee() {
    requestAnimationFrame(animYee);
    yeeCellControls.update();
    yeeCellRenderer.render(yeeCellScene, yeeCellCamera);
  }
  animYee();
}

function buildYeeCellGeometry() {
  // ── Wireframe cube ──
  const box  = new THREE.BoxGeometry(1, 1, 1);
  const edges = new THREE.EdgesGeometry(box);
  yeeCellScene.add(new THREE.LineSegments(edges,
    new THREE.LineBasicMaterial({ color: 0x1a2a4a, linewidth: 1 })));

  // ── Semi-transparent face planes ──
  const faceMat = new THREE.MeshBasicMaterial({
    color: 0x0d1a30, opacity: 0.18, transparent: true, side: THREE.DoubleSide
  });
  [
    // XY faces (z ± 0.5)
    [0,0,-0.5, 0, 0, 0],
    [0,0, 0.5, 0, 0, 0],
    // YZ faces (x ± 0.5)
    [-0.5,0,0, 0, Math.PI/2, 0],
    [ 0.5,0,0, 0, Math.PI/2, 0],
    // XZ faces (y ± 0.5)
    [0,-0.5,0, Math.PI/2, 0, 0],
    [0, 0.5,0, Math.PI/2, 0, 0],
  ].forEach(([px,py,pz, rx,ry,rz]) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1,1), faceMat);
    m.position.set(px,py,pz);
    m.rotation.set(rx,ry,rz);
    yeeCellScene.add(m);
  });

  const CYAN    = 0x00d4ff;
  const MAGENTA = 0xff1a6e;
  const arrowLen = 0.32;

  // ── E-field arrows (cyan) — at edge midpoints ──
  // Ex on 4 x-parallel edges
  [[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5,0.5]].forEach(([y,z]) =>
    yeeCellScene.add(makeArrow([0,y,z], [1,0,0], arrowLen, CYAN)));
  // Ey on 4 y-parallel edges
  [[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5,0.5]].forEach(([x,z]) =>
    yeeCellScene.add(makeArrow([x,0,z], [0,1,0], arrowLen, CYAN)));
  // Ez on 4 z-parallel edges
  [[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5,0.5]].forEach(([x,y]) =>
    yeeCellScene.add(makeArrow([x,y,0], [0,0,1], arrowLen, CYAN)));

  // ── H-field arrows (magenta) — at face centers ──
  // Hx on YZ faces → aligned with x at face center
  [[-0.5,0,0],[0.5,0,0]].forEach(([x,y,z]) =>
    yeeCellScene.add(makeArrow([x,y,z], [1,0,0], arrowLen*0.85, MAGENTA)));
  // Hy on XZ faces
  [[0,-0.5,0],[0,0.5,0]].forEach(([x,y,z]) =>
    yeeCellScene.add(makeArrow([x,y,z], [0,1,0], arrowLen*0.85, MAGENTA)));
  // Hz on XY faces
  [[0,0,-0.5],[0,0,0.5]].forEach(([x,y,z]) =>
    yeeCellScene.add(makeArrow([x,y,z], [0,0,1], arrowLen*0.85, MAGENTA)));

  // ── Position dots ──
  addDots(CYAN,    [
    [0,-0.5,-0.5],[0,0.5,-0.5],[0,-0.5,0.5],[0,0.5,0.5],  // Ex
    [-0.5,0,-0.5],[0.5,0,-0.5],[-0.5,0,0.5],[0.5,0,0.5],  // Ey
    [-0.5,-0.5,0],[0.5,-0.5,0],[-0.5,0.5,0],[0.5,0.5,0],  // Ez
  ], 0.038);
  addDots(MAGENTA, [
    [-0.5,0,0],[0.5,0,0],  // Hx
    [0,-0.5,0],[0,0.5,0],  // Hy
    [0,0,-0.5],[0,0,0.5],  // Hz
  ], 0.046);
}

function makeArrow(origin, dir, length, color) {
  const o = new THREE.Vector3(...origin);
  const d = new THREE.Vector3(...dir).normalize();
  // Center the arrow at origin
  const start = o.clone().addScaledVector(d, -length / 2);
  const arrow = new THREE.ArrowHelper(d, start, length, color, length * 0.28, length * 0.14);
  return arrow;
}

function addDots(color, positions, radius) {
  const geom = new THREE.SphereGeometry(radius, 10, 10);
  const mat  = new THREE.MeshBasicMaterial({ color });
  positions.forEach(([x,y,z]) => {
    const dot = new THREE.Mesh(geom, mat);
    dot.position.set(x, y, z);
    yeeCellScene.add(dot);
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  THREE.JS — HERO BACKGROUND ANIMATION
// ══════════════════════════════════════════════════════════════════════════════

function initHeroAnimation() {
  const container = document.getElementById('hero-bg');
  if (!container) return;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x050810, 1);
  container.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 6);

  // Expanding ring waves
  const rings = [];
  for (let i = 0; i < 12; i++) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.1, 0.015, 12, 80),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(`hsl(${185 + i * 12}, 100%, 55%)`), transparent: true, opacity: 0 })
    );
    ring.userData = { phase: i * (1.0 / 12) };
    ring.rotation.x = Math.random() * 0.8 - 0.4;
    ring.rotation.y = Math.random() * 0.8 - 0.4;
    scene.add(ring);
    rings.push(ring);
  }

  // Central glow sphere
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 20, 20),
    new THREE.MeshBasicMaterial({ color: 0x00d4ff })
  );
  scene.add(glow);

  // Star field
  const starGeom = new THREE.BufferGeometry();
  const starPos  = new Float32Array(1200 * 3);
  for (let i = 0; i < starPos.length; i++) starPos[i] = (Math.random() - 0.5) * 30;
  starGeom.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeom,
    new THREE.PointsMaterial({ color: 0xffffff, size: 0.025, transparent: true, opacity: 0.35 })));

  // Second set of tilted rings (3D feel)
  const rings2 = [];
  for (let i = 0; i < 8; i++) {
    const r = new THREE.Mesh(
      new THREE.TorusGeometry(0.1, 0.012, 10, 64),
      new THREE.MeshBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: 0 })
    );
    r.userData = { phase: i * (1.0 / 8) + 0.5 };
    r.rotation.x = Math.PI / 3 + Math.random() * 0.5;
    r.rotation.z = Math.random() * Math.PI;
    scene.add(r);
    rings2.push(r);
  }

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function tick() {
    requestAnimationFrame(tick);
    const t = Date.now() * 0.001;

    rings.forEach(r => {
      const p = ((t * 0.38 + r.userData.phase) % 1.0);
      const s = 0.3 + p * 4.5;
      r.scale.setScalar(s);
      r.material.opacity = Math.max(0, (1.0 - p) * 0.7);
    });
    rings2.forEach(r => {
      const p = ((t * 0.28 + r.userData.phase) % 1.0);
      const s = 0.3 + p * 3.5;
      r.scale.setScalar(s);
      r.material.opacity = Math.max(0, (1.0 - p) * 0.4);
    });

    glow.scale.setScalar(1.0 + Math.sin(t * 3.5) * 0.12);
    camera.position.x = Math.sin(t * 0.08) * 0.6;
    camera.position.y = Math.cos(t * 0.11) * 0.35;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
  }
  tick();
}

// ══════════════════════════════════════════════════════════════════════════════
//  UI CONTROLS
// ══════════════════════════════════════════════════════════════════════════════

function initControls() {

  // ── Playback ──
  const btnPlay  = document.getElementById('btn-play');
  const btnStep  = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');

  if (btnPlay)  btnPlay.addEventListener('click', togglePlay);
  if (btnStep)  btnStep.addEventListener('click', singleStep);
  if (btnReset) btnReset.addEventListener('click', resetSim);

  // Steps per frame
  const spfSel = document.getElementById('steps-per-frame');
  if (spfSel) spfSel.addEventListener('change', e => {
    state.stepsPerFrame = parseInt(e.target.value);
  });

  // ── Grid size ──
  const gridSel = document.getElementById('grid-size');
  if (gridSel) gridSel.addEventListener('change', e => {
    state.gridKey    = e.target.value;
    state.gridConfig = { ...GRID_CONFIGS[e.target.value] };
    state.running    = false;
    updatePlayBtn();
    // Rebuild scene + restart worker
    rebuildSlicePlanes();
    initWorker();
  });

  // ── Source controls ──
  const freqSlider = document.getElementById('source-freq');
  const ampSlider  = document.getElementById('source-amplitude');

  if (freqSlider) freqSlider.addEventListener('input', e => {
    document.getElementById('source-freq-val').textContent = `${parseFloat(e.target.value).toFixed(2)}×`;
  });
  if (ampSlider) ampSlider.addEventListener('input', e => {
    document.getElementById('source-amp-val').textContent = parseFloat(e.target.value).toFixed(1);
  });

  const applySourceBtn = document.getElementById('btn-apply-source');
  if (applySourceBtn) applySourceBtn.addEventListener('click', applySourceConfig);

  // ── Slice sliders ──
  ['x', 'y', 'z'].forEach(axis => {
    const slider = document.getElementById(`slice-${axis}`);
    if (!slider) return;
    slider.addEventListener('input', e => {
      const val = parseInt(e.target.value);
      state.sliceIdx[axis] = val;

      // Move slice plane
      const cfg = state.gridConfig;
      if (axis === 'x' && sliceMeshes.yz) sliceMeshes.yz.position.x = val / cfg.Nx - 0.5;
      if (axis === 'y' && sliceMeshes.xz) sliceMeshes.xz.position.y = val / cfg.Ny - 0.5;
      if (axis === 'z' && sliceMeshes.xy) sliceMeshes.xy.position.z = val / cfg.Nz - 0.5;

      // Tell worker
      if (worker) worker.postMessage({ type: 'setSlice', data: { axis, value: val } });
    });
  });

  // ── Material presets ──
  document.querySelectorAll('[data-material]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-material]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeMaterial = btn.dataset.material;
      applyMaterialPreset(btn.dataset.material);
    });
  });

  // ── Experiments ──
  document.querySelectorAll('[data-experiment]').forEach(btn => {
    btn.addEventListener('click', () => loadExperiment(btn.dataset.experiment));
  });

  // ── Display toggles ──
  const togXY  = document.getElementById('tog-xy');
  const togXZ  = document.getElementById('tog-xz');
  const togYZ  = document.getElementById('tog-yz');
  const togBox = document.getElementById('tog-box');
  const togAxes = document.getElementById('tog-axes');

  if (togXY)  togXY.addEventListener('change',  e => { if (sliceMeshes.xy)  sliceMeshes.xy.visible  = e.target.checked; });
  if (togXZ)  togXZ.addEventListener('change',  e => { if (sliceMeshes.xz)  sliceMeshes.xz.visible  = e.target.checked; });
  if (togYZ)  togYZ.addEventListener('change',  e => { if (sliceMeshes.yz)  sliceMeshes.yz.visible  = e.target.checked; });
  if (togBox) togBox.addEventListener('change', e => { if (boxMesh)          boxMesh.visible          = e.target.checked; });
  if (togAxes) togAxes.addEventListener('change', e => { if (axesHelper)     axesHelper.visible       = e.target.checked; });

  // ── FFT button ──
  const fftBtn = document.getElementById('btn-fft');
  if (fftBtn) fftBtn.addEventListener('click', computeAndDrawFFT);
}

function togglePlay() {
  state.running = !state.running;
  updatePlayBtn();
  if (state.running && state.workerReady) scheduleNextStep();
}

function updatePlayBtn() {
  const btn = document.getElementById('btn-play');
  if (!btn) return;
  btn.textContent = state.running ? '⏸ Pause' : '▶ Play';
  btn.classList.toggle('active', state.running);
}

function singleStep() {
  if (!state.workerReady) return;
  worker.postMessage({ type: 'step', data: { stepsPerFrame: 1 } });
}

function resetSim() {
  state.running = false;
  updatePlayBtn();
  if (worker) worker.postMessage({ type: 'reset' });
}

function applySourceConfig() {
  if (!worker || !state.workerReady) return;
  const { Nx, Ny, Nz, dx } = state.gridConfig;

  const type     = document.getElementById('source-type')?.value     || 'point';
  const waveform = document.getElementById('source-waveform')?.value  || 'ricker';
  const freqMult = parseFloat(document.getElementById('source-freq')?.value) || 1.0;
  const amplitude= parseFloat(document.getElementById('source-amplitude')?.value) || 1.0;

  const baseFreq = C0 / (20 * dx);

  const src = {
    type, waveform,
    i: type === 'plane' ? Math.floor(Nx * 0.15) : Math.floor(Nx / 2),
    j: Math.floor(Ny / 2),
    k: Math.floor(Nz / 2),
    component: 'Ez',
    freq: baseFreq * freqMult,
    amplitude,
    t0: 30,
    tw: 10,
    halfLen: 6,
    axis: 'x',
  };

  worker.postMessage({ type: 'setSources', data: [src] });
}

// ══════════════════════════════════════════════════════════════════════════════
//  MATERIAL PRESETS
// ══════════════════════════════════════════════════════════════════════════════

function applyMaterialPreset(preset) {
  if (!worker || !state.workerReady) return;
  const { Nx, Ny, Nz } = state.gridConfig;

  // Helper to send a paint command
  const paint = (x0,y0,z0, x1,y1,z1, eps, sig) =>
    worker.postMessage({ type: 'paint', data: { x0,y0,z0,x1,y1,z1,eps,sig } });

  // Reset to vacuum first
  paint(0,0,0, Nx-1,Ny-1,Nz-1, 1.0, 0.0);

  const SIG_PEC = 1e7;   // Very high conductivity ~ PEC

  switch (preset) {
    case 'free_space':
      // Already reset — nothing more needed
      break;

    case 'dielectric_slab': {
      const x0 = Math.floor(Nx * 0.55);
      const x1 = Math.floor(Nx * 0.68);
      paint(x0, 0, 0, x1, Ny-1, Nz-1, 4.0, 0.0);
      break;
    }

    case 'double_slit': {
      const wx  = Math.floor(Nx * 0.5);
      const s1y = Math.floor(Ny * 0.33);
      const s2y = Math.floor(Ny * 0.67);
      const sw  = Math.max(2, Math.floor(Ny * 0.07));
      // Full wall
      paint(wx, 0, 0, wx+1, Ny-1, Nz-1, 1.0, SIG_PEC);
      // Slit 1
      paint(wx, s1y-sw, 0, wx+1, s1y+sw, Nz-1, 1.0, 0.0);
      // Slit 2
      paint(wx, s2y-sw, 0, wx+1, s2y+sw, Nz-1, 1.0, 0.0);
      break;
    }

    case 'pec_plate': {
      const px = Math.floor(Nx / 2);
      paint(px, Math.floor(Ny*0.2), 0, px+1, Math.floor(Ny*0.8), Nz-1, 1.0, SIG_PEC);
      break;
    }

    case 'waveguide': {
      const y0 = Math.floor(Ny * 0.28);
      const y1 = Math.floor(Ny * 0.72);
      const z0 = Math.floor(Nz * 0.28);
      const z1 = Math.floor(Nz * 0.72);
      // Four metallic walls
      paint(0, y0, z0, Nx-1, y0+1, z1, 1.0, SIG_PEC);
      paint(0, y1-1, z0, Nx-1, y1, z1, 1.0, SIG_PEC);
      paint(0, y0, z0, Nx-1, y1, z0+1, 1.0, SIG_PEC);
      paint(0, y0, z1-1, Nx-1, y1, z1, 1.0, SIG_PEC);
      break;
    }

    case 'cavity': {
      const cx0 = Math.floor(Nx*0.3), cx1 = Math.floor(Nx*0.7);
      const cy0 = Math.floor(Ny*0.3), cy1 = Math.floor(Ny*0.7);
      const cz0 = Math.floor(Nz*0.3), cz1 = Math.floor(Nz*0.7);
      // 6 walls
      paint(cx0, cy0, cz0, cx0+1, cy1, cz1, 1.0, SIG_PEC);
      paint(cx1-1, cy0, cz0, cx1, cy1, cz1, 1.0, SIG_PEC);
      paint(cx0, cy0, cz0, cx1, cy0+1, cz1, 1.0, SIG_PEC);
      paint(cx0, cy1-1, cz0, cx1, cy1, cz1, 1.0, SIG_PEC);
      paint(cx0, cy0, cz0, cx1, cy1, cz0+1, 1.0, SIG_PEC);
      paint(cx0, cy0, cz1-1, cx1, cy1, cz1, 1.0, SIG_PEC);
      break;
    }

    default: break;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  GUIDED EXPERIMENTS
// ══════════════════════════════════════════════════════════════════════════════

function loadExperiment(name) {
  // Stop simulation, reset, apply preset, navigate to simulator
  state.running = false;
  updatePlayBtn();
  if (worker) worker.postMessage({ type: 'reset' });

  const { Nx, Ny, Nz, dx } = state.gridConfig;
  const baseFreq = C0 / (20 * dx);

  setTimeout(() => {
    // Apply material first
    const materialMap = {
      free_space:   'free_space',
      diffraction:  'double_slit',
      dielectric:   'dielectric_slab',
      cavity:       'cavity',
      waveguide:    'waveguide',
      dipole:       'free_space',
    };
    applyMaterialPreset(materialMap[name] || 'free_space');

    // Configure source
    let src = {
      type: 'point', component: 'Ez', waveform: 'ricker',
      i: Math.floor(Nx/2), j: Math.floor(Ny/2), k: Math.floor(Nz/2),
      freq: baseFreq, amplitude: 1.0, t0: 30, tw: 10, halfLen: 6
    };

    switch (name) {
      case 'diffraction':
        src = { ...src, type: 'plane', waveform: 'sinusoidal',
          i: Math.floor(Nx * 0.12), freq: baseFreq * 0.8 };
        break;
      case 'dielectric':
        src = { ...src, type: 'plane', waveform: 'ricker',
          i: Math.floor(Nx * 0.12) };
        break;
      case 'cavity':
        src = { ...src, waveform: 'gaussian',
          i: Math.floor(Nx*0.42), j: Math.floor(Ny*0.42), k: Math.floor(Nz*0.42),
          t0: 20, tw: 5 };
        break;
      case 'waveguide':
        src = { ...src, type: 'plane', waveform: 'sinusoidal',
          i: Math.floor(Nx * 0.1), freq: baseFreq * 1.2 };
        break;
      case 'dipole':
        src = { ...src, type: 'dipole', waveform: 'sinusoidal',
          freq: baseFreq * 0.9 };
        break;
    }

    worker.postMessage({ type: 'setSources', data: [src] });

    // Navigate to simulator and auto-start
    document.getElementById('simulator')?.scrollIntoView({ behavior: 'smooth' });

    setTimeout(() => {
      state.running = true;
      updatePlayBtn();
      scheduleNextStep();
    }, 600);
  }, 200);
}

// ══════════════════════════════════════════════════════════════════════════════
//  PROBE CHART
// ══════════════════════════════════════════════════════════════════════════════

function drawProbeChart() {
  // Both probe-canvas (sidebar) and td-canvas (analysis section)
  drawSignalOnCanvas('probe-canvas', state.probeHistory, 0x00d4ff);
  drawSignalOnCanvas('td-canvas',    state.probeHistory, 0x00d4ff);
}

function clearTDChart() {
  clearCanvas('td-canvas');
  clearCanvas('probe-canvas');
  clearCanvas('fft-canvas');
}

function clearCanvas(id) {
  const c = document.getElementById(id);
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, c.width, c.height);
}

function drawSignalOnCanvas(canvasId, data, colorHex) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || data.length < 2) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Find range
  let maxAbs = 1e-12;
  data.forEach(v => { if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v); });

  // Center line
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  // Signal
  const r = (colorHex >> 16) & 0xff;
  const g = (colorHex >>  8) & 0xff;
  const b =  colorHex        & 0xff;

  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = 1.5;
  ctx.shadowColor = `rgb(${r},${g},${b})`;
  ctx.shadowBlur = 4;
  ctx.beginPath();

  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H / 2 - (v / maxAbs) * (H / 2 - 4);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });

  ctx.stroke();
  ctx.shadowBlur = 0;
}

// ══════════════════════════════════════════════════════════════════════════════
//  FFT ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════

function computeAndDrawFFT() {
  const data = state.tdHistory;
  if (data.length < 16) return;

  // Zero-pad to next power of 2
  const M = nextPow2(data.length);
  const re = new Float64Array(M);
  const im = new Float64Array(M);
  for (let i = 0; i < data.length; i++) re[i] = data[i];

  cooleyTukeyFFT(re, im, M);

  const half = M / 2;
  let maxMag = 1e-30;
  const mags = new Float32Array(half);
  for (let i = 1; i < half; i++) {
    mags[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    if (mags[i] > maxMag) maxMag = mags[i];
  }

  // Draw
  const canvas = document.getElementById('fft-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Gradient bars
  const grad = ctx.createLinearGradient(0, H, 0, 0);
  grad.addColorStop(0.0, '#7c3aed');
  grad.addColorStop(0.5, '#00a8d0');
  grad.addColorStop(1.0, '#00d4ff');
  ctx.fillStyle = grad;

  const bw = W / half;
  for (let i = 1; i < half; i++) {
    const x  = (i / half) * W;
    const bh = (mags[i] / maxMag) * H;
    ctx.fillRect(x, H - bh, Math.max(1, bw - 0.5), bh);
  }

  // Glow layer
  ctx.globalAlpha = 0.3;
  ctx.shadowColor = '#00d4ff';
  ctx.shadowBlur  = 8;
  ctx.fillStyle   = '#00d4ff';
  for (let i = 1; i < half; i++) {
    const x  = (i / half) * W;
    const bh = (mags[i] / maxMag) * H;
    ctx.fillRect(x, H - bh, Math.max(1, bw - 0.5), 2);
  }
  ctx.globalAlpha = 1;
  ctx.shadowBlur  = 0;
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** Iterative Cooley-Tukey FFT (in-place, complex input) */
function cooleyTukeyFFT(re, im, N) {
  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // Butterfly
  for (let len = 2; len <= N; len <<= 1) {
    const ang  = -2 * Math.PI / len;
    const wRe  = Math.cos(ang);
    const wIm  = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len >> 1; k++) {
        const uRe = re[i + k],        uIm = im[i + k];
        const vRe = re[i+k+len/2] * curRe - im[i+k+len/2] * curIm;
        const vIm = re[i+k+len/2] * curIm + im[i+k+len/2] * curRe;
        re[i + k]         = uRe + vRe;
        im[i + k]         = uIm + vIm;
        re[i + k + len/2] = uRe - vRe;
        im[i + k + len/2] = uIm - vIm;
        const nRe = curRe * wRe - curIm * wIm;
        const nIm = curRe * wIm + curIm * wRe;
        curRe = nRe; curIm = nIm;
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  ALGORITHM WALKTHROUGH
// ══════════════════════════════════════════════════════════════════════════════

const ALGO_STEPS = [
  {
    title: 'Step 1 — Initialize Fields',
    equation: 'E^0 = \\mathbf{0}, \\quad H^{-1/2} = \\mathbf{0}',
    code:
`// All six field components start at zero.
Ex = new Float32Array(Nx * Ny * Nz).fill(0);
Ey = new Float32Array(Nx * Ny * Nz).fill(0);
Ez = new Float32Array(Nx * Ny * Nz).fill(0);
Hx = new Float32Array(Nx * Ny * Nz).fill(0);
Hy = new Float32Array(Nx * Ny * Nz).fill(0);
Hz = new Float32Array(Nx * Ny * Nz).fill(0);`,
    desc: 'The simulation begins from perfect electromagnetic silence. All six field components (Ex, Ey, Ez — electric; Hx, Hy, Hz — magnetic) are set to zero everywhere in the 3D grid.'
  },
  {
    title: 'Step 2 — Compute E-update Coefficients',
    equation: 'C_a = \\frac{\\varepsilon - \\sigma\\Delta t/2}{\\varepsilon + \\sigma\\Delta t/2}, \\quad C_b = \\frac{\\Delta t/\\Delta x}{\\varepsilon + \\sigma\\Delta t/2}',
    code:
`// Pre-compute per-cell material coefficients.
for (let n = 0; n < N; n++) {
  const eps   = epsr[n] * EPS0;     // permittivity
  const half  = sigmaE[n] * dt / 2; // σΔt/2
  const denom = eps + half;
  Ca[n] = (eps - half) / denom;     // loss factor
  Cb[n] = (dt / dx)    / denom;     // coupling
}`,
    desc: 'Ca and Cb encode the material properties of every cell: permittivity (ε) controls how strongly E fields form; conductivity (σ) introduces loss that damps fields over time. Cells with high σ act as absorbers (like the PML boundary).'
  },
  {
    title: 'Step 3 — Leapfrog: Update H Fields (½ step)',
    equation: `H^{n+\\frac{1}{2}} = H^{n-\\frac{1}{2}} - \\frac{\\Delta t}{\\mu_0 \\Delta x}\\nabla \\times E^n`,
    code:
`// Hx: −(1/μ₀)(∂Ez/∂y − ∂Ey/∂z)
for (let i=0; i<Nx-1; i++) {
  for (let j=0; j<Ny-1; j++) {
    for (let k=0; k<Nz-1; k++) {
      const n = i*NyNz + j*Nz + k;
      Hx[n] -= dtmu * ((Ez[n+Nz] - Ez[n])
                      - (Ey[n+1]  - Ey[n]));
      Hy[n] -= dtmu * ((Ex[n+1]   - Ex[n])
                      - (Ez[n+NyNz] - Ez[n]));
      Hz[n] -= dtmu * ((Ey[n+NyNz] - Ey[n])
                      - (Ex[n+Nz]  - Ex[n]));
    }
  }
}`,
    desc: 'The H field is pushed forward by half a time step. The discrete curl of E is computed using differences of four neighbouring E-field values — no integrals, just subtraction. This is the "leapfrog" jump: H is advanced to time n+½ while E stays at time n.'
  },
  {
    title: 'Step 4 — Inject Source',
    equation: 'E_z[i_s, j_s, k_s] \\mathrel{+}= A \\cdot f(n\\Delta t)',
    code:
`// Ricker wavelet ("Mexican hat") — inherently broadband
const ts  = t0 * dt;           // peak time
const tnow = timeStep * dt;    // current time
const u    = Math.PI * fc * (tnow - ts);
const val  = (1 - 2*u*u) * Math.exp(-u*u);
Ez[srcIdx] += amplitude * val; // additive soft source`,
    desc: 'A "soft source" additively injects the excitation waveform into the grid — it does not force the field to a specific value, so back-scattered waves can pass through the source point transparently. The Ricker wavelet is ideal: it has a known, smooth broadband spectrum.'
  },
  {
    title: 'Step 5 — Leapfrog: Update E Fields (full step)',
    equation: 'E^{n+1} = C_a \\cdot E^n + C_b \\cdot \\nabla \\times H^{n+\\frac{1}{2}}',
    code:
`// Ez: (1/ε)(∂Hy/∂x − ∂Hx/∂y)
for (let i=1; i<Nx-1; i++) {
  for (let j=1; j<Ny-1; j++) {
    for (let k=1; k<Nz-1; k++) {
      const n  = i*NyNz + j*Nz + k;
      Ez[n] = Ca[n]*Ez[n] + Cb[n] * (
        (Hy[n] - Hy[n-NyNz]) -   // ∂Hy/∂x
        (Hx[n] - Hx[n-Nz])       // ∂Hx/∂y
      );
    }
  }
}`,
    desc: 'The electric E field is driven forward by the updated H field. Ca introduces loss; Cb converts the spatial H-curl into a field update. The symmetry is perfect: E drives H in Step 3, and H drives E here — the wave self-propagates!'
  },
  {
    title: 'Step 6 — Apply PML Boundary Absorption',
    equation: 'E, H \\xrightarrow{\\text{PML}} E \\cdot e^{-\\sigma_{\\text{PML}} \\Delta t / \\varepsilon_0},\\; H \\cdot e^{-\\sigma_{\\text{PML}} \\Delta t / \\varepsilon_0}',
    code:
`// Decay factor precomputed at init (cubic grading)
// pmlDecay[n] = exp(-σ_PML * dt / ε₀)
// = 1.0 in the interior, < 1.0 at boundaries

for (let n = 0; n < N; n++) {
  const f = pmlDecay[n];     // ≈ 1.0 for interior cells
  Ex[n]*=f; Ey[n]*=f; Ez[n]*=f;
  Hx[n]*=f; Hy[n]*=f; Hz[n]*=f;
}`,
    desc: 'The Perfectly Matched Layer (PML) absorbs outgoing waves at the grid boundary to simulate infinite open space. Fields in the PML region are exponentially attenuated each time step. The cubic conductivity grading prevents reflections at the PML interface.'
  },
  {
    title: 'Step 7 — Repeat Forever',
    equation: 'n \\leftarrow n+1, \\quad \\text{goto Step 3}',
    code:
`// Main simulation loop (runs in a Web Worker)
while (running) {
  updateH();          // Step 3: H^{n-½} → H^{n+½}
  injectSources();    // Step 4: add excitation
  updateE();          // Step 5: E^n  → E^{n+1}
  applyPML();         // Step 6: absorb at boundaries
  extractSlices();    // send 3 cross-sections to UI
  timeStep++;
}`,
    desc: 'That is the entire algorithm! Each time step costs O(N) operations on a grid of N cells — fully explicit, no matrix solve required. The wave propagates forward in time autonomously from the leapfrog coupling between E and H.'
  }
];

let algoStep = 0;

function initAlgorithmWalkthrough() {
  renderAlgoStep(0);

  document.getElementById('algo-prev')?.addEventListener('click', () => {
    algoStep = Math.max(0, algoStep - 1);
    renderAlgoStep(algoStep);
  });
  document.getElementById('algo-next')?.addEventListener('click', () => {
    algoStep = Math.min(ALGO_STEPS.length - 1, algoStep + 1);
    renderAlgoStep(algoStep);
  });
}

function renderAlgoStep(idx) {
  const step = ALGO_STEPS[idx];
  const container = document.getElementById('algo-content');
  if (!container || !step) return;

  container.innerHTML = `
    <div class="algo-step">
      <div class="algo-header">
        <span class="algo-num">${idx + 1} / ${ALGO_STEPS.length}</span>
        <h3>${step.title}</h3>
      </div>
      <div class="algo-equation">\\[${step.equation}\\]</div>
      <div class="algo-code"><pre><code>${escHtml(step.code)}</code></pre></div>
      <div class="algo-description">${step.desc}</div>
      <div class="algo-progress">
        ${ALGO_STEPS.map((_, i) =>
          `<div class="algo-dot ${i === idx ? 'active' : i < idx ? 'done' : ''}"></div>`
        ).join('')}
      </div>
    </div>`;

  // Re-render MathJax in this container
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise([container]).catch(() => {});
  }

  const prev = document.getElementById('algo-prev');
  const next = document.getElementById('algo-next');
  if (prev) prev.disabled = idx === 0;
  if (next) next.disabled = idx === ALGO_STEPS.length - 1;
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ══════════════════════════════════════════════════════════════════════════════
//  NAVIGATION SCROLLSPY
// ══════════════════════════════════════════════════════════════════════════════

function initScrollspy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => {
          a.classList.toggle('active', a.getAttribute('href') === `#${entry.target.id}`);
        });
      }
    });
  }, { threshold: 0.25, rootMargin: `-${64}px 0px -40% 0px` });

  sections.forEach(s => obs.observe(s));

  // Smooth-scroll all anchor links
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  BOOT
// ══════════════════════════════════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
  initScrollspy();
  initHeroAnimation();
  initSimScene();
  initYeeCellScene();
  initWorker();
  initControls();
  initAlgorithmWalkthrough();

  // Trigger MathJax on the whole page once
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise().catch(() => {});
  }
});
