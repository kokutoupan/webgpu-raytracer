// src/main.ts
import { WebGPURenderer } from './renderer';
import { WorldBridge } from './world-bridge';
import * as WebMMuxer from 'webm-muxer'; // WebM (VP9)

// --- DOM Elements ---
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement; 
const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement;
const inputWidth = document.getElementById('res-width') as HTMLInputElement;
const inputHeight = document.getElementById('res-height') as HTMLInputElement;
const inputFile = document.getElementById('obj-file') as HTMLInputElement;
if (inputFile) inputFile.accept = ".obj,.glb,.vrm";

const inputDepth = document.getElementById('max-depth') as HTMLInputElement;
const inputSPP = document.getElementById('spp-frame') as HTMLInputElement;
const btnRecompile = document.getElementById('recompile-btn') as HTMLButtonElement;
const inputUpdateInterval = document.getElementById('update-interval') as HTMLInputElement;
const animSelect = document.getElementById('anim-select') as HTMLSelectElement;

// 録画用UI
const btnRecord = document.getElementById('record-btn') as HTMLButtonElement;
const inputRecFps = document.getElementById('rec-fps') as HTMLInputElement;
const inputRecDur = document.getElementById('rec-duration') as HTMLInputElement;
const inputRecSPP = document.getElementById('rec-spp') as HTMLInputElement;
const inputRecBatch = document.getElementById('rec-batch') as HTMLInputElement;

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
let isRecording = false; 
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
    renderer.updateScreenSize(w, h); // This also resets accumulation
    
    // Camera update is part of uniform update now, will happen in render loop or explicit call
    if (worldBridge.hasWorld) {
        worldBridge.updateCamera(w, h);
        renderer.updateSceneUniforms(worldBridge.cameraData, 0);
    }
    renderer.recreateBindGroup();
    renderer.resetAccumulation();
    frameCount = 0;
    totalFrameCount = 0;
  };

  const loadScene = async (name: string, autoStart = true) => {
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

    await renderer.loadTexturesFromWorld(worldBridge);

    // Initial Buffer Upload
    renderer.updateCombinedGeometry(worldBridge.vertices, worldBridge.normals, worldBridge.uvs);
    renderer.updateCombinedBVH(worldBridge.tlas, worldBridge.blas);
    
    renderer.updateBuffer('index', worldBridge.indices);
    renderer.updateBuffer('attr', worldBridge.attributes);
    renderer.updateBuffer('instance', worldBridge.instances);

    updateResolution(); // This triggers recreateBindGroup
    updateAnimList();

    if (autoStart) {
      isRendering = true;
      if (btn) btn.textContent = "Stop Rendering";
    }
  };

  // 録画機能
  const recordVideo = async () => {
    if (isRecording) return;

    isRendering = false;
    isRecording = true;

    btnRecord.textContent = "Initializing...";
    btnRecord.disabled = true;
    if (btn) btn.textContent = "Resume Rendering"; 

    const fps = parseInt(inputRecFps.value, 10) || 30;
    const duration = parseInt(inputRecDur.value, 10) || 3;
    const totalFrames = fps * duration;

    const samplesPerFrame = parseInt(inputRecSPP.value, 10) || 64;
    const batchSize = parseInt(inputRecBatch.value, 10) || 4;

    console.log(`Starting recording: ${totalFrames} frames @ ${fps}fps (VP9)`);

    const muxer = new WebMMuxer.Muxer({
      target: new WebMMuxer.ArrayBufferTarget(),
      video: {
        codec: 'V_VP9',
        width: canvas.width,
        height: canvas.height,
        frameRate: fps
      }
    });

    const videoEncoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => console.error("VideoEncoder Error:", e)
    });

    videoEncoder.configure({
      codec: 'vp09.00.10.08', 
      width: canvas.width,
      height: canvas.height,
      bitrate: 12_000_000,
    });

    try {
      for (let i = 0; i < totalFrames; i++) {
        btnRecord.textContent = `Rec: ${i}/${totalFrames} (${Math.round(i / totalFrames * 100)}%)`;

        await new Promise(r => setTimeout(r, 0));

        const time = i / fps;
        worldBridge.update(time);

        // 2. ジオメトリ更新
        let needsRebind = false;
        needsRebind ||= renderer.updateCombinedBVH(worldBridge.tlas, worldBridge.blas);
        needsRebind ||= renderer.updateBuffer('instance', worldBridge.instances);
        
        // Skinning support: Vertices/Normals need update every frame
        needsRebind ||= renderer.updateCombinedGeometry(worldBridge.vertices, worldBridge.normals, worldBridge.uvs);
        
        // BVH Rebuild sorts triangles, so indices and attributes change order every frame!
        needsRebind ||= renderer.updateBuffer('index', worldBridge.indices);
        needsRebind ||= renderer.updateBuffer('attr', worldBridge.attributes);

        worldBridge.updateCamera(canvas.width, canvas.height); // Ensure camera matches
        renderer.updateSceneUniforms(worldBridge.cameraData, 0);

        if (needsRebind) renderer.recreateBindGroup();
        renderer.resetAccumulation();

        // 3. 分割レンダリング
        let samplesDone = 0;
        while (samplesDone < samplesPerFrame) {
          const batch = Math.min(batchSize, samplesPerFrame - samplesDone);
          for (let k = 0; k < batch; k++) {
             renderer.render(samplesDone + k);
          }
          samplesDone += batch;
          await renderer.device.queue.onSubmittedWorkDone();
          if (samplesDone < samplesPerFrame) {
            await new Promise(r => setTimeout(r, 0));
          }
        }

        if (videoEncoder.encodeQueueSize > 5) {
          await videoEncoder.flush();
        }

        const frame = new VideoFrame(canvas, {
          timestamp: (i * 1000000) / fps,
          duration: 1000000 / fps
        });

        videoEncoder.encode(frame, { keyFrame: i % fps === 0 });
        frame.close();
      }

      btnRecord.textContent = "Finalizing...";
      await videoEncoder.flush();
      muxer.finalize();

      const { buffer } = muxer.target;
      const blob = new Blob([buffer], { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `raytrace_${Date.now()}.webm`;
      a.click();
      URL.revokeObjectURL(url);

    } catch (e) {
      console.error("Recording failed:", e);
      alert("Recording failed. See console.");
    } finally {
      isRecording = false;
      isRendering = true;
      btnRecord.textContent = "● Rec";
      btnRecord.disabled = false;
      if (btn) btn.textContent = "Stop Rendering";
      requestAnimationFrame(renderFrame);
    }
  };

  // --- Render Loop ---
  let lastTime = performance.now();
  let frameTimer = 0;

  const renderFrame = () => {
    if (isRecording) return; 

    requestAnimationFrame(renderFrame);
    if (!isRendering || !worldBridge.hasWorld) return;

    let updateInterval = parseInt(inputUpdateInterval.value, 10);
    if (isNaN(updateInterval) || updateInterval < 0) updateInterval = 0;

    if (updateInterval > 0 && frameCount >= updateInterval) {
      worldBridge.update(totalFrameCount / updateInterval / 60);

      let needsRebind = false;
      needsRebind ||= renderer.updateCombinedBVH(worldBridge.tlas, worldBridge.blas);
      needsRebind ||= renderer.updateBuffer('instance', worldBridge.instances);
      needsRebind ||= renderer.updateCombinedGeometry(worldBridge.vertices, worldBridge.normals, worldBridge.uvs);
      
      // BVH Rebuild sorts triangles, so indices and attributes change order every frame!
      needsRebind ||= renderer.updateBuffer('index', worldBridge.indices);
      needsRebind ||= renderer.updateBuffer('attr', worldBridge.attributes);
      
      worldBridge.updateCamera(canvas.width, canvas.height);
      renderer.updateSceneUniforms(worldBridge.cameraData, 0);

      if (needsRebind) {
        renderer.recreateBindGroup();
      }
      renderer.resetAccumulation();
      frameCount = 0; 
    } else {
        // Even if no update, we might want camera updates if we had controls (mouse orbit etc not impl yet but good practice)
        // For now, static camera.
    }

    frameCount++;
    frameTimer++;
    totalFrameCount++;
    
    // Update uniforms (camera, frame)
    // Actually we should separate camera update from animation update
    // But user controls camera implicitly via WASM update currently if WASM has camera controls?
    // The WASM `update_camera` is mostly for aspect ratio. 
    // Let's ensure uniforms are up to date.
    // Optimization: Only update if changed? 
    // Always updating is safer for now.
    
    // We update sceneUniforms with frameCount. But wait, renderer.render() also updates frameCount in uniforms.
    // renderer.updateSceneUniforms(worldBridge.cameraData, frameCount); 
    // The renderer.render() call handles frameCount partial update. 
    // But we need to ensure camera data is there.
    
    // renderer.updateSceneUniforms should only be called if camera changes or BLAS offset changes.
    // We already called it during update interval or load.
    
    renderer.render(frameCount);

    const now = performance.now();
    if (now - lastTime >= 1000) {
      statsDiv.textContent = `FPS: ${frameTimer} | ${(1000 / frameTimer).toFixed(2)}ms | Frame: ${frameCount}`;
      frameTimer = 0;
      lastTime = now;
    }
  };

  // --- Event Listeners ---
  if (btn) {
    btn.addEventListener("click", () => {
      isRendering = !isRendering;
      btn.textContent = isRendering ? "Stop Rendering" : "Resume Rendering";
    });
  }

  if (btnRecord) {
    btnRecord.addEventListener("click", recordVideo);
  }

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

  const updateAnimList = () => {
    const list = worldBridge.getAnimationList();
    animSelect.innerHTML = "";
    if (list.length === 0) {
      const opt = document.createElement("option");
      opt.text = "No Anim";
      animSelect.add(opt);
      animSelect.disabled = true;
      return;
    }
    animSelect.disabled = false;
    list.forEach((name, i) => {
      const opt = document.createElement("option");
      opt.text = `[${i}] ${name}`;
      opt.value = i.toString();
      animSelect.add(opt);
    });
    animSelect.value = "0";
  };

  animSelect.addEventListener("change", () => {
    const idx = parseInt(animSelect.value, 10);
    worldBridge.setAnimation(idx);
  });

  updateResolution();
  loadScene("cornell", false);
  requestAnimationFrame(renderFrame);
}

main().catch(console.error);
