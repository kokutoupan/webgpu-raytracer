import shaderCode from './shader.wgsl?raw';

// DOM取得
const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
const btn = document.getElementById('render-btn') as HTMLButtonElement;


// --- ベクトル演算用の簡易ヘルパー ---
const vec3 = {
  create: (x: number, y: number, z: number) => ({ x, y, z }),
  sub: (a: any, b: any) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }),
  add: (a: any, b: any) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }),
  scale: (v: any, s: number) => ({ x: v.x * s, y: v.y * s, z: v.z * s }),
  cross: (a: any, b: any) => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  }),
  normalize: (v: any) => {
    const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len === 0 ? { x: 0, y: 0, z: 0 } : { x: v.x / len, y: v.y / len, z: v.z / len };
  },
  len: (v: any) => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),
};

// --- ランダム・カラー生成ヘルパー ---
const rnd = () => Math.random();
const rndRange = (min: number, max: number) => min + (max - min) * Math.random();

// --- カメラデータ生成関数 ---
// "One Weekend" シリーズと同じパラメータを受け取ります
function createCameraData(
  lookfrom: { x: number; y: number; z: number },
  lookat: { x: number; y: number; z: number },
  vup: { x: number; y: number; z: number },
  vfov: number,        // 垂直画角 (度)
  aspectRatio: number,
  defocusAngle: number,// ボケの強さ (0ならピンホール)
  focusDist: number    // ピントが合う距離
): Float32Array<ArrayBuffer> {
  const theta = (vfov * Math.PI) / 180.0;
  const h = Math.tan(theta / 2.0);
  const viewportHeight = 2.0 * h * focusDist;
  const viewportWidth = viewportHeight * aspectRatio;

  // 座標系の基底ベクトルを計算 (w: 奥, u: 右, v: 上)
  const w = vec3.normalize(vec3.sub(lookfrom, lookat));
  const u = vec3.normalize(vec3.cross(vup, w));
  const v = vec3.cross(w, u);

  // カメラのパラメータ
  const origin = lookfrom;
  const horizontal = vec3.scale(u, viewportWidth);
  const vertical = vec3.scale(v, viewportHeight);

  // lower_left_corner = origin - horizontal/2 - vertical/2 - focusDist*w
  const lowerLeftCorner = vec3.sub(
    vec3.sub(vec3.sub(origin, vec3.scale(horizontal, 0.5)), vec3.scale(vertical, 0.5)),
    vec3.scale(w, focusDist)
  );

  // レンズ半径 (defocus_angle / 2)
  const lensRadius = focusDist * Math.tan((defocusAngle * Math.PI) / 360.0);

  // GPU送信用配列 (要素数: 24, バイト数: 96)
  // alignment: vec3 は 16byte (4 floats) 境界に配置するのが安全
  // [Origin(3), LensRadius(1), LowerLeft(3), pad(1), Horizontal(3), pad(1), Vertical(3), pad(1), u(3), pad(1), v(3), pad(1)]
  return new Float32Array([
    origin.x, origin.y, origin.z, lensRadius,         // offset 0
    lowerLeftCorner.x, lowerLeftCorner.y, lowerLeftCorner.z, 0.0, // offset 16
    horizontal.x, horizontal.y, horizontal.z, 0.0,    // offset 32
    vertical.x, vertical.y, vertical.z, 0.0,          // offset 48
    u.x, u.y, u.z, 0.0,                               // offset 64 (Defocus用)
    v.x, v.y, v.z, 0.0                                // offset 80 (Defocus用)
  ]);
}


// --- マテリアル定義 (読みやすくするため) ---
const MatType = {
  Lambertian: 0.0,
  Metal: 1.0,
  Dielectric: 2.0,
};

// --- 球データ生成ヘルパー ---
// 戻り値: GPU用フォーマットに合わせた number[] (要素数12)
function createSphere(
  center: { x: number, y: number, z: number },
  radius: number,
  color: { r: number, g: number, b: number },
  matType: number,
  extra: number = 0.0 // Fuzz(金属) や IOR(ガラス)
): number[] {
  return [
    // 0-15 bytes: Center(vec3) + Radius(f32)
    center.x, center.y, center.z, radius,

    // 16-31 bytes: Color(vec3) + MatType(f32)
    color.r, color.g, color.b, matType,

    // 32-47 bytes: Extra(f32) + Padding(3*f32)
    extra, 0.0, 0.0, 0.0
  ];
}

function makeSpheres(){
    // --- 球データの作成 ---
  const spheresList: number[][] = [];

  // 1. 地面 (巨大な球)
  spheresList.push(createSphere(
    { x: 0.0, y: -1000.0, z: 0.0 }, 1000.0,
    { r: 0.5, g: 0.5, b: 0.5 }, MatType.Lambertian
  ));

  // 2. ランダムな小球 (-11 to 11 の範囲)
  for (let a = -11; a < 11; a++) {
    for (let b = -11; b < 11; b++) {
      const chooseMat = rnd();
      const center = {
        x: a + 0.9 * rnd(),
        y: 0.2,
        z: b + 0.9 * rnd()
      };

      // point3(4, 0.2, 0) との距離チェック
      const dist = vec3.len(vec3.sub(center, { x: 4.0, y: 0.2, z: 0.0 }));

      if (dist > 0.9) {
        if (chooseMat < 0.8) {
          // diffuse: color * color (成分ごとの積)
          const r = rnd() * rnd();
          const g = rnd() * rnd();
          const b = rnd() * rnd();
          spheresList.push(createSphere(
            center, 0.2,
            { r, g, b }, MatType.Lambertian
          ));
        } else if (chooseMat < 0.95) {
          // metal: albedo random(0.5, 1), fuzz random(0, 0.5)
          const r = rndRange(0.5, 1.0);
          const g = rndRange(0.5, 1.0);
          const b = rndRange(0.5, 1.0);
          const fuzz = rndRange(0.0, 0.5);
          spheresList.push(createSphere(
            center, 0.2,
            { r, g, b }, MatType.Metal,
            fuzz
          ));
        } else {
          // glass
          spheresList.push(createSphere(
            center, 0.2,
            { r: 1.0, g: 1.0, b: 1.0 }, MatType.Dielectric,
            1.5
          ));
        }
      }
    }
  }

  // 3. 大きな球 (Glass)
  spheresList.push(createSphere(
    { x: 0.0, y: 1.0, z: 0.0 }, 1.0,
    { r: 1.0, g: 1.0, b: 1.0 }, MatType.Dielectric,
    1.5
  ));

  // 4. 大きな球 (Lambertian)
  spheresList.push(createSphere(
    { x: -4.0, y: 1.0, z: 0.0 }, 1.0,
    { r: 0.4, g: 0.2, b: 0.1 }, MatType.Lambertian
  ));

  // 5. 大きな球 (Metal)
  spheresList.push(createSphere(
    { x: 4.0, y: 1.0, z: 0.0 }, 1.0,
    { r: 0.7, g: 0.6, b: 0.5 }, MatType.Metal,
    0.0
  ));

  return spheresList;
}


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
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- 3. カメラパラメータの計算と送信 (CPU側) ---
  {
    // ★ ここで自由にカメラを設定できます
    const cam = {
      lookfrom: { x: 13., y: 2.0, z: 3.0 },
      lookat: { x: 0.0, y: 0.0, z: 0.0 },
      vup: { x: 0.0, y: 1.0, z: 0.0 },
      vfov: 20.0,
      aspectRatio: canvas.width / canvas.height,
      defocusAngle: 0.2, // 0.0 にするとボケなし
      focusDist: 10,
    };

    const cameraData = createCameraData(
      cam.lookfrom, cam.lookat, cam.vup,
      cam.vfov, cam.aspectRatio,
      cam.defocusAngle, cam.focusDist
    );

    device.queue.writeBuffer(cameraUniformBuffer, 0, cameraData);
  }

  // --- 球データの作成 (ヘルパー使用) ---
  // 配列の配列を作って、最後に flat() で1次元にします
  const spheresList = makeSpheres();
  // const spheresList: number[][] = [];
  //
  // // 1. 地面 (巨大な緑の球)
  // spheresList.push(createSphere(
  //   { x: 0.0, y: -100.5, z: -1.0 }, 100.0,
  //   { r: 0.8, g: 0.8, b: 0.0 }, MatType.Lambertian
  // ));
  //
  // // 2. 中央 (青い拡散球)
  // spheresList.push(createSphere(
  //   { x: 0.0, y: 0.0, z: -1.0 }, 0.5,
  //   { r: 0.1, g: 0.2, b: 0.5 }, MatType.Lambertian
  // ));
  //
  // // 3. 左 (ガラス球)
  // spheresList.push(createSphere(
  //   { x: -1.0, y: 0.0, z: -1.0 }, 0.5,
  //   { r: 1.0, g: 1.0, b: 1.0 }, MatType.Dielectric,
  //   1.5 // 屈折率
  // ));
  // // 3. 左 (ガラス球)
  // spheresList.push(createSphere(
  //   { x: -1.0, y: 0.0, z: -1.0 }, 0.4,
  //   { r: 1.0, g: 1.0, b: 1.0 }, MatType.Dielectric,
  //   1.0/1.5 // 屈折率
  // ));
  //
  // // 4. 右 (金色の金属球)
  // spheresList.push(createSphere(
  //   { x: 1.0, y: 0.0, z: -1.0 }, 0.5,
  //   { r: 0.8, g: 0.6, b: 0.2 }, MatType.Metal,
  //   0.0 // Fuzz
  // ));

  // --- GPU送信用データへの変換 ---
  // [ [sphere1...], [sphere2...] ] -> [sphere1..., sphere2...]
  const flattenedData = spheresList.flat();
  const sphereF32Array = new Float32Array(flattenedData);

  // 球バッファ作成
  const sphereBuffer = device.createBuffer({
    size: sphereF32Array.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // データを書き込む
  device.queue.writeBuffer(sphereBuffer, 0, sphereF32Array);

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
        { binding: 4, resource: { buffer: sphereBuffer } }, // 球データ
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

  // --- イベントハンドラ (修正版) ---
  btn.addEventListener("click", () => {
    if (isRendering) {
      // ■ 動作中 -> 停止
      console.log("Stop Rendering");
      isRendering = false;
      btn.textContent = "Restart Rendering"; // ボタンの文字を変えると分かりやすい
    } else {
      // ■ 停止中 -> リセットして再開
      console.log("Reset & Start Rendering");

      // 1. カウンタリセット
      frameCount = 0;

      // 2. 蓄積バッファのクリア (黒で埋める)
      // bufferSize はバイト数なので、Float32Arrayの要素数は 1/4
      const zeroData = new Float32Array(bufferSize / 4);
      // fill(0)はデフォルト値なので明示しなくても0だが、念のため
      device.queue.writeBuffer(accumulateBuffer, 0, zeroData);

      // 3. レンダリング開始
      isRendering = true;
      renderFrame();
      btn.textContent = "Stop Rendering";
    }
  });
}

initAndRender().catch(console.error);
