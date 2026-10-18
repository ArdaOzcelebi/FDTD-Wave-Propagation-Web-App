/**
 * fdtd_worker.js — 3D FDTD Electromagnetic Simulation Engine
 *
 * Implements the Yee-lattice FDTD algorithm for 3D Maxwell's equations.
 * Runs inside a Web Worker to keep the UI thread responsive.
 *
 * Fields on the Yee grid (isotropic, dx=dy=dz):
 *   Hx[i,j,k] = −(1/μ₀)(∂Ez/∂y − ∂Ey/∂z)
 *   Hy[i,j,k] = −(1/μ₀)(∂Ex/∂z − ∂Ez/∂x)
 *   Hz[i,j,k] = −(1/μ₀)(∂Ey/∂x − ∂Ex/∂y)
 *   Ex[i,j,k] = Ca·Ex + Cb·(∂Hz/∂y − ∂Hy/∂z)
 *   Ey[i,j,k] = Ca·Ey + Cb·(∂Hx/∂z − ∂Hz/∂x)
 *   Ez[i,j,k] = Ca·Ez + Cb·(∂Hy/∂x − ∂Hx/∂y)
 *
 * Index convention: n = i*Ny*Nz + j*Nz + k
 *   n+1    → k+1 (z direction)
 *   n+Nz   → j+1 (y direction)
 *   n+NyNz → i+1 (x direction)
 */

'use strict';

// ─── Physical Constants ───────────────────────────────────────────────────────
const MU0  = 4.0 * Math.PI * 1e-7;          // H/m  — permeability of free space
const EPS0 = 8.854187817e-12;               // F/m  — permittivity of free space
const C0   = 1.0 / Math.sqrt(MU0 * EPS0);  // m/s  — speed of light ≈ 2.998×10⁸

// ─── Grid & Field State ───────────────────────────────────────────────────────
let Nx, Ny, Nz, NyNz, N;
let dx, dt;

let Ex, Ey, Ez;   // Electric field components  [V/m]
let Hx, Hy, Hz;   // Magnetic field components  [A/m]
let Ca, Cb;        // E-update material coefficients
let epsr, sigmaE;  // Relative permittivity, electric conductivity [S/m]
let pmlDecay;      // Pre-computed PML decay factors (1.0 in interior)

// ─── Simulation State ─────────────────────────────────────────────────────────
let timeStep   = 0;
let initialized = false;
let sources    = [];
let sliceX, sliceY, sliceZ;

// PML depth (cells on each face)
const PML_D = 10;

// ─── Message Router ───────────────────────────────────────────────────────────
self.onmessage = function ({ data: msg }) {
  switch (msg.type) {
    case 'init':      handleInit(msg.data);    break;
    case 'step':      handleStep(msg.data);    break;
    case 'paint':     handlePaint(msg.data);   break;
    case 'setSlice':  handleSetSlice(msg.data); break;
    case 'setSources':sources = msg.data || []; break;
    case 'reset':     handleReset();            break;
    case 'getProbe':  handleProbe(msg.data);    break;
    default: break;
  }
};

// ─── Initialization ───────────────────────────────────────────────────────────
function handleInit(cfg) {
  Nx    = cfg.Nx  || 60;
  Ny    = cfg.Ny  || 60;
  Nz    = cfg.Nz  || 60;
  NyNz  = Ny * Nz;
  N     = Nx * NyNz;
  dx    = cfg.dx  || 3e-3;   // cell size [m], default 3 mm
  // Courant stability: dt ≤ dx / (c √3)
  dt    = 0.99 * dx / (C0 * Math.sqrt(3.0));

  // Allocate field arrays
  Ex = new Float32Array(N);  Ey = new Float32Array(N);  Ez = new Float32Array(N);
  Hx = new Float32Array(N);  Hy = new Float32Array(N);  Hz = new Float32Array(N);

  // Material arrays — default: vacuum
  epsr   = new Float32Array(N).fill(1.0);
  sigmaE = new Float32Array(N).fill(0.0);

  // Update coefficient arrays
  Ca = new Float32Array(N);
  Cb = new Float32Array(N);
  computeCoefficients();

  // PML graded-decay array
  pmlDecay = new Float32Array(N);
  buildPMLDecay();

  // Default slice positions (grid center)
  sliceX = (cfg.sliceX != null) ? cfg.sliceX : Math.floor(Nx / 2);
  sliceY = (cfg.sliceY != null) ? cfg.sliceY : Math.floor(Ny / 2);
  sliceZ = (cfg.sliceZ != null) ? cfg.sliceZ : Math.floor(Nz / 2);

  // Install sources
  if (cfg.sources && cfg.sources.length > 0) {
    sources = cfg.sources;
  } else {
    sources = [defaultSource()];
  }

  timeStep    = 0;
  initialized = true;

  // Notify main thread
  const Courant = C0 * dt / dx * Math.sqrt(3.0);
  self.postMessage({
    type: 'ready',
    data: { Nx, Ny, Nz, dx, dt, C0, Courant,
            freqCenter: C0 / (20 * dx) }
  });
}

function defaultSource() {
  return {
    type      : 'point',
    i         : Math.floor(Nx / 2),
    j         : Math.floor(Ny / 2),
    k         : Math.floor(Nz / 2),
    component : 'Ez',
    waveform  : 'ricker',
    freq      : C0 / (20 * dx),   // wavelength = 20 cells
    amplitude : 1.0,
    t0        : 30,                // time offset [steps]
    tw        : 10                 // pulse width [steps]
  };
}

// ─── Material Coefficients ────────────────────────────────────────────────────
/**
 * Ca = (ε − σΔt/2) / (ε + σΔt/2)
 * Cb = (Δt / (ε + σΔt/2)) / dx
 */
function computeCoefficients() {
  for (let n = 0; n < N; n++) {
    const eps   = epsr[n] * EPS0;
    const sig   = sigmaE[n];
    const half  = sig * dt * 0.5;
    const denom = eps + half;
    Ca[n] = (eps - half) / denom;
    Cb[n] = (dt / dx)    / denom;
  }
}

// ─── PML Construction ─────────────────────────────────────────────────────────
/**
 * Cubic-polynomial grading: σ(d) ∝ (d/D)³
 * translated into a per-cell decay factor applied after each leapfrog step.
 * Factor = 1 in the interior, exponentially less at boundaries.
 */
function buildPMLDecay() {
  const sigMax = 0.8 * (PML_D + 1) / (MU0 * C0 * dx * PML_D);

  for (let i = 0; i < Nx; i++) {
    for (let j = 0; j < Ny; j++) {
      for (let k = 0; k < Nz; k++) {
        const di = Math.min(i, Nx - 1 - i);
        const dj = Math.min(j, Ny - 1 - j);
        const dk = Math.min(k, Nz - 1 - k);
        const d  = Math.min(di, dj, dk);
        const n  = i * NyNz + j * Nz + k;

        if (d < PML_D) {
          const rho  = (PML_D - d) / PML_D;           // 0 at inner edge, 1 at boundary
          const sig  = sigMax * rho * rho * rho;       // cubic grading
          pmlDecay[n] = Math.exp(-sig * dt / EPS0);   // exact exponential decay
        } else {
          pmlDecay[n] = 1.0;
        }
      }
    }
  }
}

// ─── Time Stepping ────────────────────────────────────────────────────────────
function handleStep({ stepsPerFrame = 3 } = {}) {
  if (!initialized) return;

  for (let s = 0; s < stepsPerFrame; s++) {
    updateH();
    injectSources();   // inject before E update so source drives H next half-step
    updateE();
    applyPML();
    timeStep++;
  }

  sendFrame();
}

/** H-field leapfrog: H^{n+½} = H^{n-½} − (Δt/μ₀) ∇×E^n */
function updateH() {
  const dtmu = dt / (MU0 * dx);

  for (let i = 0; i < Nx - 1; i++) {
    const iBase = i * NyNz;
    for (let j = 0; j < Ny - 1; j++) {
      let base = iBase + j * Nz;
      for (let k = 0; k < Nz - 1; k++, base++) {
        // Hx: −(1/μ)(∂Ez/∂y − ∂Ey/∂z)
        Hx[base] -= dtmu * ((Ez[base + Nz] - Ez[base]) - (Ey[base + 1] - Ey[base]));
        // Hy: −(1/μ)(∂Ex/∂z − ∂Ez/∂x)
        Hy[base] -= dtmu * ((Ex[base + 1] - Ex[base]) - (Ez[base + NyNz] - Ez[base]));
        // Hz: −(1/μ)(∂Ey/∂x − ∂Ex/∂y)
        Hz[base] -= dtmu * ((Ey[base + NyNz] - Ey[base]) - (Ex[base + Nz] - Ex[base]));
      }
    }
  }
}

/** E-field leapfrog: E^{n+1} = Ca·E^n + Cb·∇×H^{n+½} */
function updateE() {
  for (let i = 1; i < Nx - 1; i++) {
    const iBase = i * NyNz;
    for (let j = 1; j < Ny - 1; j++) {
      let base = iBase + j * Nz + 1;
      for (let k = 1; k < Nz - 1; k++, base++) {
        const ca = Ca[base];
        const cb = Cb[base];
        // Ex: Ca·Ex + Cb·(∂Hz/∂y − ∂Hy/∂z)
        Ex[base] = ca * Ex[base] + cb * ((Hz[base] - Hz[base - Nz])   - (Hy[base] - Hy[base - 1]));
        // Ey: Ca·Ey + Cb·(∂Hx/∂z − ∂Hz/∂x)
        Ey[base] = ca * Ey[base] + cb * ((Hx[base] - Hx[base - 1])   - (Hz[base] - Hz[base - NyNz]));
        // Ez: Ca·Ez + Cb·(∂Hy/∂x − ∂Hx/∂y)
        Ez[base] = ca * Ez[base] + cb * ((Hy[base] - Hy[base - NyNz]) - (Hx[base] - Hx[base - Nz]));
      }
    }
  }
}

// ─── Source Injection ─────────────────────────────────────────────────────────
function injectSources() {
  const t = timeStep;

  for (const src of sources) {
    const val = excitation(src, t);
    if (val === 0) continue;

    const amp = src.amplitude * val;

    if (src.type === 'point') {
      const n = src.i * NyNz + src.j * Nz + src.k;
      if (n >= 0 && n < N) injectComponent(n, src.component, amp);

    } else if (src.type === 'plane') {
      // Plane-wave injection across a YZ plane at x = src.i
      const xi = src.i;
      if (xi < 0 || xi >= Nx) continue;
      for (let j = PML_D; j < Ny - PML_D; j++) {
        for (let k = PML_D; k < Nz - PML_D; k++) {
          injectComponent(xi * NyNz + j * Nz + k, src.component, amp);
        }
      }

    } else if (src.type === 'dipole') {
      // Short z-directed dipole along a line in z at (src.i, src.j)
      const halfLen = src.halfLen || 5;
      for (let k = src.k - halfLen; k <= src.k + halfLen; k++) {
        if (k < PML_D || k >= Nz - PML_D) continue;
        injectComponent(src.i * NyNz + src.j * Nz + k, src.component, amp);
      }
    }
  }
}

function injectComponent(n, component, amp) {
  switch (component) {
    case 'Ex': Ex[n] += amp; break;
    case 'Ey': Ey[n] += amp; break;
    case 'Ez': default: Ez[n] += amp; break;
  }
}

function excitation(src, t) {
  const fc  = src.freq;
  const t0  = src.t0  || 30;
  const tw  = src.tw  || 10;

  switch (src.waveform) {
    case 'sinusoidal':
      return Math.sin(2.0 * Math.PI * fc * t * dt);

    case 'gaussian': {
      return Math.exp(-Math.pow((t - t0) / tw, 2));
    }

    case 'ricker': {
      // Mexican-hat (Ricker) wavelet — inherently broadband
      const ts   = t0 * dt;
      const tNow = t  * dt;
      const u    = Math.PI * fc * (tNow - ts);
      return (1.0 - 2.0 * u * u) * Math.exp(-u * u);
    }

    case 'modulated_gaussian': {
      const env = Math.exp(-Math.pow((t - t0) / tw, 2));
      return env * Math.sin(2.0 * Math.PI * fc * t * dt);
    }

    default:
      return Math.sin(2.0 * Math.PI * fc * t * dt);
  }
}

// ─── PML Application ─────────────────────────────────────────────────────────
/**
 * Multiply all six field components by the pre-computed decay factor.
 * Interior cells have pmlDecay = 1.0, so this is a no-op there.
 */
function applyPML() {
  for (let n = 0; n < N; n++) {
    const f = pmlDecay[n];
    Ex[n] *= f;  Ey[n] *= f;  Ez[n] *= f;
    Hx[n] *= f;  Hy[n] *= f;  Hz[n] *= f;
  }
}

// ─── Frame Extraction & Transmission ─────────────────────────────────────────
function sendFrame() {
  // Allocate fresh buffers each frame (ownership will be transferred)
  const xySlice = new Float32Array(Nx * Ny);   // z = sliceZ plane
  const xzSlice = new Float32Array(Nx * Nz);   // y = sliceY plane
  const yzSlice = new Float32Array(Ny * Nz);   // x = sliceX plane

  for (let i = 0; i < Nx; i++) {
    for (let j = 0; j < Ny; j++) {
      xySlice[j * Nx + i] = Ez[i * NyNz + j * Nz + sliceZ];
    }
  }
  for (let i = 0; i < Nx; i++) {
    for (let k = 0; k < Nz; k++) {
      xzSlice[k * Nx + i] = Ez[i * NyNz + sliceY * Nz + k];
    }
  }
  for (let j = 0; j < Ny; j++) {
    for (let k = 0; k < Nz; k++) {
      yzSlice[k * Ny + j] = Ez[sliceX * NyNz + j * Nz + k];
    }
  }

  // Running max |E| for colormap normalization
  let maxE = 1e-20;
  for (let n = 0; n < N; n++) {
    const e2 = Ex[n] * Ex[n] + Ey[n] * Ey[n] + Ez[n] * Ez[n];
    if (e2 > maxE) maxE = e2;
  }
  maxE = Math.sqrt(maxE);

  // Probe value at slice intersection
  const probeEz = Ez[sliceX * NyNz + sliceY * Nz + sliceZ];

  self.postMessage(
    {
      type: 'frame',
      data: { xySlice, xzSlice, yzSlice,
              maxE: Math.max(maxE, 1e-12),
              timeStep, time: timeStep * dt,
              probeEz }
    },
    // Transfer ownership of the underlying ArrayBuffers — zero-copy!
    [xySlice.buffer, xzSlice.buffer, yzSlice.buffer]
  );
}

// ─── Material Painting ────────────────────────────────────────────────────────
function handlePaint({ x0, y0, z0, x1, y1, z1, eps, sig }) {
  const xi0 = Math.max(0, Math.min(Nx - 1, x0 | 0));
  const xi1 = Math.max(0, Math.min(Nx - 1, x1 | 0));
  const yi0 = Math.max(0, Math.min(Ny - 1, y0 | 0));
  const yi1 = Math.max(0, Math.min(Ny - 1, y1 | 0));
  const zi0 = Math.max(0, Math.min(Nz - 1, z0 | 0));
  const zi1 = Math.max(0, Math.min(Nz - 1, z1 | 0));

  for (let i = xi0; i <= xi1; i++) {
    for (let j = yi0; j <= yi1; j++) {
      for (let k = zi0; k <= zi1; k++) {
        const n   = i * NyNz + j * Nz + k;
        epsr[n]   = eps;
        sigmaE[n] = sig;
      }
    }
  }
  computeCoefficients();
}

// ─── Slice Position ───────────────────────────────────────────────────────────
function handleSetSlice({ axis, value }) {
  if (axis === 'x') sliceX = Math.max(0, Math.min(Nx - 1, value | 0));
  if (axis === 'y') sliceY = Math.max(0, Math.min(Ny - 1, value | 0));
  if (axis === 'z') sliceZ = Math.max(0, Math.min(Nz - 1, value | 0));
}

// ─── Reset ────────────────────────────────────────────────────────────────────
function handleReset() {
  if (!initialized) return;
  Ex.fill(0); Ey.fill(0); Ez.fill(0);
  Hx.fill(0); Hy.fill(0); Hz.fill(0);
  // Reset materials to vacuum
  epsr.fill(1.0);
  sigmaE.fill(0.0);
  computeCoefficients();
  timeStep = 0;
  self.postMessage({ type: 'reset_done' });
}

// ─── Point Probe ─────────────────────────────────────────────────────────────
function handleProbe({ i, j, k }) {
  if (!initialized) return;
  const n = i * NyNz + j * Nz + k;
  self.postMessage({
    type: 'probe_data',
    data: { Ex: Ex[n], Ey: Ey[n], Ez: Ez[n],
            Hx: Hx[n], Hy: Hy[n], Hz: Hz[n],
            t: timeStep * dt }
  });
}
