import shaderCode from './shader.wgsl?raw';

// DOM取得
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;

// WebGPUの初期化とレンダリング
async function initAndRender() {
  // --- 1. WebGPU初期化 ---
  if (!navigator.gpu) { alert("WebGPU not supported."); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { throw new Error("No adapter found"); }
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) { throw new Error("WebGPU context not found"); }

  // const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING
  });

  // --- 2. リソース作成 (Buffer & Texture) ---

  // A. 蓄積用バッファ (Accumulation Buffer)
  // ピクセル数 * RGBA(4) * float32(4byte)
  const bufferSize = canvas.width * canvas.height * 4 * 4;
  const accumulateBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // B. フレーム情報 Uniform (毎フレーム更新)
  // frame_count (u32) + padding (3 * u32) = 16 bytes (一応アライメントを気にして確保)
  const frameUniformBuffer = device.createBuffer({
    size: 16, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // C. カメラ情報 Uniform (初期化時のみ更新)
  // 以下の4つのvec3を持つ (WGSLのvec3は16byteアライメント推奨)
  // origin(16) + lower_left(16) + horizontal(16) + vertical(16) = 64 bytes
  const cameraUniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- 3. カメラパラメータの計算と送信 (CPU側) ---
  {
    const aspectRatio = canvas.width / canvas.height;
    const viewportHeight = 2.0;
    const viewportWidth = aspectRatio * viewportHeight;
    const focalLength = 1.0;

    const origin = { x: 0.0, y: 0.0, z: 0.0 };
    const horizontal = { x: viewportWidth, y: 0.0, z: 0.0 };
    const vertical = { x: 0.0, y: viewportHeight, z: 0.0 };
    
    // lower_left_corner = origin - horizontal/2 - vertical/2 - (0,0,focal_length)
    const lowerLeftCorner = {
      x: origin.x - horizontal.x / 2.0 - vertical.x / 2.0 - 0.0,
      y: origin.y - horizontal.y / 2.0 - vertical.y / 2.0 - 0.0,
      z: origin.z - horizontal.z / 2.0 - vertical.z / 2.0 - focalLength,
    };

    // Float32Arrayにパックする (vec3はパディングを入れて4要素にする)
    const cameraData = new Float32Array([
      origin.x, origin.y, origin.z, 0.0,            // origin
      lowerLeftCorner.x, lowerLeftCorner.y, lowerLeftCorner.z, 0.0, // lower_left
      horizontal.x, horizontal.y, horizontal.z, 0.0,    // horizontal
      vertical.x, vertical.y, vertical.z, 0.0,        // vertical
    ]);

    // GPUに書き込み (ループの外で1回だけ！)
    device.queue.writeBuffer(cameraUniformBuffer, 0, cameraData);
  }

  // --- 4. パイプライン作成 ---
  const shaderModule = device.createShaderModule({ label: "RayTracing", code: shaderCode });
  const pipeline = device.createComputePipeline({
    label: "Main Pipeline",
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });

  // ★最適化: ループ内で使う定数やオブジェクトをキャッシュしておく
  const bindGroupLayout = pipeline.getBindGroupLayout(0);
  const frameData = new Uint32Array(1); // データ転送用の配列を使い回す

  // --- 5. レンダリングループ ---
  let frameCount = 0;
  let isRendering = false;

  const renderFrame = () => {
    if (!isRendering) return;

    frameCount++;

    // フレーム番号のみ更新 (配列再利用)
    frameData[0] = frameCount;
    device.queue.writeBuffer(frameUniformBuffer, 0, frameData);

    const texture = context.getCurrentTexture();
    const textureView = texture.createView();

    // BindGroup作成
    // ※ 出力先テクスチャ(textureView)が毎フレーム変わるため、BindGroupの再生成は必須
    // (さらに最適化するにはシェーダーでBindGroupを分けて、変化しないリソースを再利用する方法がある)
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout, // ★キャッシュを使用
      entries: [
        { binding: 0, resource: textureView },                    // Output Texture
        { binding: 1, resource: { buffer: accumulateBuffer } },   // Accumulation Buffer
        { binding: 2, resource: { buffer: frameUniformBuffer } }, // Frame Info
        { binding: 3, resource: { buffer: cameraUniformBuffer } },// Camera Info (Fixed)
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // 8x8スレッドグループでディスパッチ
    passEncoder.dispatchWorkgroups(
      Math.ceil(canvas.width / 8),
      Math.ceil(canvas.height / 8)
    );
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(renderFrame);
  };

  // --- イベントハンドラ ---
  btn.addEventListener("click", () => {
    if (!isRendering) {
      console.log("Start Rendering...");
      isRendering = true;
      renderFrame();
    } else {
      // リセットロジック (カメラを動かした場合などはここも修正が必要)
      console.log("Resetting accumulation...");
      frameCount = 0;
      // accumulateBufferの中身をゼロクリアするとより丁寧
      device.queue.writeBuffer(accumulateBuffer, 0, new Float32Array(bufferSize/4).fill(0));
    }
  });
}

initAndRender().catch(console.error);
