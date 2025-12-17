import { Config } from "../config";

export class UIManager {
  // Elements
  public canvas: HTMLCanvasElement;
  public btnRender: HTMLButtonElement;
  public sceneSelect: HTMLSelectElement;
  public inputWidth: HTMLInputElement;
  public inputHeight: HTMLInputElement;
  public inputFile: HTMLInputElement;
  public inputDepth: HTMLInputElement;
  public inputSPP: HTMLInputElement;
  public btnRecompile: HTMLButtonElement;
  public inputUpdateInterval: HTMLInputElement;
  public animSelect: HTMLSelectElement;

  public btnRecord: HTMLButtonElement;
  public inputRecFps: HTMLInputElement;
  public inputRecDur: HTMLInputElement;
  public inputRecSpp: HTMLInputElement; // Fixed type name conflict if any
  public inputRecBatch: HTMLInputElement;

  public btnHost: HTMLButtonElement;
  public btnWorker: HTMLButtonElement;
  public btnSendScene: HTMLButtonElement;
  public statusDiv: HTMLDivElement;

  public statsDiv: HTMLDivElement;

  // Callbacks
  public onRenderStart: (() => void) | null = null;
  public onRenderStop: (() => void) | null = null;
  public onSceneSelect: ((name: string) => void) | null = null;
  public onResolutionChange: ((w: number, h: number) => void) | null = null;
  public onRecompile: ((depth: number, spp: number) => void) | null = null;
  public onFileSelect: ((file: File) => void) | null = null;
  public onAnimSelect: ((index: number) => void) | null = null;

  public onRecordStart: (() => void) | null = null;

  public onConnectHost: (() => void) | null = null;
  public onConnectWorker: (() => void) | null = null;
  public onSendScene: (() => void) | null = null;

  constructor() {
    this.canvas = this.el<HTMLCanvasElement>(Config.ids.canvas);
    this.btnRender = this.el<HTMLButtonElement>(Config.ids.renderBtn);
    this.sceneSelect = this.el<HTMLSelectElement>(Config.ids.sceneSelect);
    this.inputWidth = this.el<HTMLInputElement>(Config.ids.resWidth);
    this.inputHeight = this.el<HTMLInputElement>(Config.ids.resHeight);
    this.inputFile = this.setupFileInput();
    this.inputDepth = this.el<HTMLInputElement>(Config.ids.maxDepth);
    this.inputSPP = this.el<HTMLInputElement>(Config.ids.sppFrame);
    this.btnRecompile = this.el<HTMLButtonElement>(Config.ids.recompileBtn);
    this.inputUpdateInterval = this.el<HTMLInputElement>(
      Config.ids.updateInterval
    );
    this.animSelect = this.el<HTMLSelectElement>(Config.ids.animSelect);

    this.btnRecord = this.el<HTMLButtonElement>(Config.ids.recordBtn);
    this.inputRecFps = this.el<HTMLInputElement>(Config.ids.recFps);
    this.inputRecDur = this.el<HTMLInputElement>(Config.ids.recDuration);
    this.inputRecSpp = this.el<HTMLInputElement>(Config.ids.recSpp);
    this.inputRecBatch = this.el<HTMLInputElement>(Config.ids.recBatch);

    this.btnHost = this.el<HTMLButtonElement>(Config.ids.btnHost);
    this.btnWorker = this.el<HTMLButtonElement>(Config.ids.btnWorker);
    this.btnSendScene = this.el<HTMLButtonElement>(Config.ids.btnSendScene);
    this.statusDiv = this.el<HTMLDivElement>(Config.ids.statusDiv);

    this.statsDiv = this.createStatsDiv();

    this.bindEvents();
  }

  private el<T extends HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Element not found: ${id}`);
    return el as T;
  }

  private setupFileInput(): HTMLInputElement {
    const el = this.el<HTMLInputElement>(Config.ids.objFile);
    if (el) el.accept = ".obj,.glb,.vrm";
    return el;
  }

  private createStatsDiv(): HTMLDivElement {
    const div = document.createElement("div");
    Object.assign(div.style, {
      position: "fixed",
      bottom: "10px",
      left: "10px",
      color: "#0f0",
      background: "rgba(0,0,0,0.7)",
      padding: "8px",
      fontFamily: "monospace",
      fontSize: "14px",
      pointerEvents: "none",
      zIndex: "9999",
      borderRadius: "4px",
    });
    document.body.appendChild(div);
    return div;
  }

  private bindEvents() {
    this.btnRender.addEventListener("click", () => {
      if (
        this.btnRender.textContent === "Render Start" ||
        this.btnRender.textContent === "Resume Rendering"
      ) {
        this.onRenderStart?.();
        this.updateRenderButton(true);
      } else {
        this.onRenderStop?.();
        this.updateRenderButton(false);
      }
    });

    this.sceneSelect.addEventListener("change", () =>
      this.onSceneSelect?.(this.sceneSelect.value)
    );

    const triggerRes = () =>
      this.onResolutionChange?.(
        parseInt(this.inputWidth.value) || Config.defaultWidth,
        parseInt(this.inputHeight.value) || Config.defaultHeight
      );
    this.inputWidth.addEventListener("change", triggerRes);
    this.inputHeight.addEventListener("change", triggerRes);

    this.btnRecompile.addEventListener("click", () =>
      this.onRecompile?.(
        parseInt(this.inputDepth.value) || 10,
        parseInt(this.inputSPP.value) || 1
      )
    );

    this.inputFile.addEventListener("change", (e) => {
      const f = (e.target as HTMLInputElement).files?.[0];
      if (f) this.onFileSelect?.(f);
    });

    this.animSelect.addEventListener("change", () => {
      const idx = parseInt(this.animSelect.value, 10);
      this.onAnimSelect?.(idx);
    });

    this.btnRecord.addEventListener("click", () => this.onRecordStart?.());

    this.btnHost.addEventListener("click", () => this.onConnectHost?.());
    this.btnWorker.addEventListener("click", () => this.onConnectWorker?.());
    this.btnSendScene.addEventListener("click", () => this.onSendScene?.());
  }

  // --- Public API for updates ---

  public updateRenderButton(isRendering: boolean) {
    this.btnRender.textContent = isRendering
      ? "Stop Rendering"
      : "Resume Rendering";
  }

  public updateStats(fps: number, ms: number, frame: number) {
    this.statsDiv.textContent = `FPS: ${fps} | ${ms.toFixed(
      2
    )}ms | Frame: ${frame}`;
  }

  public setStatus(msg: string) {
    this.statusDiv.textContent = msg;
  }

  public setConnectionState(role: "host" | "worker" | null) {
    if (role === "host") {
      this.btnHost.textContent = "Disconnect";
      this.btnHost.disabled = false;

      this.btnWorker.textContent = "Worker";
      this.btnWorker.disabled = true; // Can't be both

      this.btnSendScene.style.display = "inline-block";
      this.btnSendScene.disabled = true; // Wait for worker
    } else if (role === "worker") {
      this.btnHost.textContent = "Host";
      this.btnHost.disabled = true;

      this.btnWorker.textContent = "Disconnect";
      this.btnWorker.disabled = false;

      this.btnSendScene.style.display = "none";
    } else {
      // Disconnected
      this.btnHost.textContent = "Host";
      this.btnHost.disabled = false;

      this.btnWorker.textContent = "Worker";
      this.btnWorker.disabled = false;

      this.btnSendScene.style.display = "none";
      this.statusDiv.textContent = "Offline";
    }
  }

  public setSendSceneEnabled(enabled: boolean) {
    this.btnSendScene.disabled = !enabled;
  }

  public setSendSceneText(text: string) {
    this.btnSendScene.textContent = text;
  }

  public setRecordingState(isRec: boolean, progressText?: string) {
    if (isRec) {
      this.btnRecord.disabled = true;
      this.btnRecord.textContent = progressText || "Recording...";
      this.btnRender.textContent = "Resume Rendering"; // Forced stop visual
    } else {
      this.btnRecord.disabled = false;
      this.btnRecord.textContent = "● Rec";
    }
  }

  public updateAnimList(list: string[]) {
    this.animSelect.innerHTML = "";
    if (list.length === 0) {
      const opt = document.createElement("option");
      opt.text = "No Anim";
      this.animSelect.add(opt);
      this.animSelect.disabled = true;
      return;
    }
    this.animSelect.disabled = false;
    list.forEach((name, i) => {
      const opt = document.createElement("option");
      opt.text = `[${i}] ${name}`;
      opt.value = i.toString();
      this.animSelect.add(opt);
    });
    this.animSelect.value = "0"; // Default
  }

  public getRenderConfig() {
    return {
      width: parseInt(this.inputWidth.value, 10) || Config.defaultWidth,
      height: parseInt(this.inputHeight.value, 10) || Config.defaultHeight,
      fps: parseInt(this.inputRecFps.value, 10) || 30,
      duration: parseFloat(this.inputRecDur.value) || 3.0,
      spp: parseInt(this.inputRecSpp.value, 10) || 64, // Corrected property name
      batch: parseInt(this.inputRecBatch.value, 10) || 4,
      anim: parseInt(this.animSelect.value, 10) || 0,
    };
  }

  public setRenderConfig(config: {
    width: number;
    height: number;
    fps: number;
    duration: number;
    spp: number;
    batch: number;
  }) {
    this.inputWidth.value = config.width.toString();
    this.inputHeight.value = config.height.toString();
    this.inputRecFps.value = config.fps.toString();
    this.inputRecDur.value = config.duration.toString();
    this.inputRecSpp.value = config.spp.toString(); // Corrected property name
    this.inputRecBatch.value = config.batch.toString();
  }
}
