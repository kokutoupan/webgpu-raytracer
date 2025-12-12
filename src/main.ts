// src/main.ts
import { WebGPURenderer } from './renderer';
import { WorldBridge } from './world-bridge';

// --- DOM Elements ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;
const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement;
const inputWidth = document.getElementById('res-width') as HTMLInputElement;
const inputHeight = document.getElementById('res-height') as HTMLInputElement;
const inputFile = document.getElementById('obj-file') as HTMLInputElement;
if (inputFile) inputFile.accept = ".obj,.glb,.vrm";
const inputAnimFile = document.getElementById('anim-file') as HTMLInputElement;
if (inputAnimFile) inputAnimFile.accept = ".glb,.gltf";

const inputDepth = document.getElementById('max-depth') as HTMLInputElement;
const inputSPP = document.getElementById('spp-frame') as HTMLInputElement;
const btnRecompile = document.getElementById('recompile-btn') as HTMLButtonElement;
const inputUpdateInterval = document.getElementById('update-interval') as HTMLInputElement;

// --- Stats UI ---
const statsDiv = document.createElement("div");
Object.assign(statsDiv.style, {
  position: "fixed", bottom: "10px", left: "10px", color: "#0f0",
  background: "rgba(0,0,0,0.7)", padding: "8px", fontFamily: "monospace",
  fontSize: "14px", pointerEvents: "none", zIndex: "9999", borderRadius: "4px"
});
document.body.appendChild(statsDiv);

// --- Global State ---
let frameCount = 0;
let isRendering = false;
let currentFileData: string | ArrayBuffer | null = null;
let currentFileType: 'obj' | 'glb' | null = null;

async function main() {
  const renderer = new WebGPURenderer(canvas);
  const worldBridge = new WorldBridge();

  let totalFrameCount = 0;


  try {
    await renderer.init();
    await worldBridge.initWasm();
  } catch (e) {
    alert("Initialization failed: " + e);
    console.error(e);
    return;
  }

  // --- Initial Setup ---
  const rebuildPipeline = () => {
    const depth = parseInt(inputDepth.value, 10) || 10;
    const spp = parseInt(inputSPP.value, 10) || 1;
    renderer.buildPipeline(depth, spp);
  };
  rebuildPipeline();

  const updateResolution = () => {
    const w = parseInt(inputWidth.value, 10) || 720;
    const h = parseInt(inputHeight.value, 10) || 480;
    renderer.updateScreenSize(w, h);
    if (worldBridge.hasWorld) {
      worldBridge.updateCamera(w, h);
      renderer.updateCameraBuffer(worldBridge.cameraData);
    }
    renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
    totalFrameCount=0;
  };

  const loadScene = (name: string, autoStart = true) => {
    isRendering = false;
    console.log(`Loading Scene: ${name}...`);

    let objSource: string | undefined;
    let glbData: Uint8Array | undefined;

    if (name === 'viewer' && currentFileData) {
      if (currentFileType === 'obj') objSource = currentFileData as string;
      else if (currentFileType === 'glb') glbData = new Uint8Array(currentFileData as ArrayBuffer);
    }

    worldBridge.loadScene(name, objSource, glbData);
    worldBridge.printStats();

    // Initial Buffer Upload
    renderer.updateGeometryBuffer('vertex', worldBridge.vertices);
    renderer.updateGeometryBuffer('normal', worldBridge.normals);
    renderer.updateGeometryBuffer('index', worldBridge.indices);
    renderer.updateGeometryBuffer('attr', worldBridge.attributes);
    renderer.updateGeometryBuffer('tlas', worldBridge.tlas);
    renderer.updateGeometryBuffer('blas', worldBridge.blas);
    renderer.updateGeometryBuffer('instance', worldBridge.instances);

    updateResolution(); // Camera update & BindGroup creation included

    if (autoStart) {
      isRendering = true;
      btn.textContent = "Stop Rendering";
    }
  };

  // --- Render Loop ---
  let lastTime = performance.now();
  let frameTimer = 0;

  const renderFrame = () => {
    requestAnimationFrame(renderFrame);
    if (!isRendering || !worldBridge.hasWorld) return;

    let updateInterval = parseInt(inputUpdateInterval.value, 10);
    if (isNaN(updateInterval) || updateInterval < 0) updateInterval = 0;

    // アニメーション更新 (インターバル毎)
    if (updateInterval > 0 && frameCount >= updateInterval) {
      worldBridge.update(totalFrameCount/updateInterval);

      // 変更があったバッファのみ更新 (戻り値がtrueならBindGroup再生成が必要)
      let needsRebind = false;
      needsRebind ||= renderer.updateGeometryBuffer('tlas', worldBridge.tlas);
      needsRebind ||= renderer.updateGeometryBuffer('blas', worldBridge.blas);
      needsRebind ||= renderer.updateGeometryBuffer('instance', worldBridge.instances);
      needsRebind ||= renderer.updateGeometryBuffer('vertex', worldBridge.vertices);
      needsRebind ||= renderer.updateGeometryBuffer('normal', worldBridge.normals);
      needsRebind ||= renderer.updateGeometryBuffer('index', worldBridge.indices);
      needsRebind ||= renderer.updateGeometryBuffer('attr', worldBridge.attributes);

      if (needsRebind) {
        renderer.recreateBindGroup();
      }
      renderer.resetAccumulation();
      frameCount = 0;
    }

    frameCount++;
    frameTimer++;
    totalFrameCount++;

    renderer.render(frameCount);

    // Stats
    const now = performance.now();
    if (now - lastTime >= 1000) {
      statsDiv.textContent = `FPS: ${frameTimer} | ${(1000 / frameTimer).toFixed(2)}ms | Frame: ${frameCount}`;
      frameTimer = 0;
      lastTime = now;
    }
  };

  // --- Event Listeners ---
  btn.addEventListener("click", () => {
    isRendering = !isRendering;
    btn.textContent = isRendering ? "Stop Rendering" : "Resume Rendering";
  });

  sceneSelect.addEventListener("change", (e) => loadScene((e.target as HTMLSelectElement).value, false));

  inputWidth.addEventListener("change", updateResolution);
  inputHeight.addEventListener("change", updateResolution);

  btnRecompile.addEventListener("click", () => {
    isRendering = false;
    rebuildPipeline();
    renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
    isRendering = true;
  });

  inputFile.addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (ext === 'obj') {
      currentFileData = await f.text();
      currentFileType = 'obj';
    } else {
      currentFileData = await f.arrayBuffer();
      currentFileType = 'glb';
    }
    sceneSelect.value = "viewer";
    loadScene("viewer", false);
  });

  inputAnimFile.addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    console.log(`Loading Motion: ${f.name}...`);
    const buffer = await f.arrayBuffer();
    worldBridge.loadAnimation(new Uint8Array(buffer));
    console.log("Motion Loaded!");
    (e.target as HTMLInputElement).value = "";
  });

  // Start
  updateResolution();
  loadScene("cornell", false);
  requestAnimationFrame(renderFrame);
}

main().catch(console.error);
