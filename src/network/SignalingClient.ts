import { Config } from "../config";
import { RtcClient } from "./RtcClient";
import type { SignalingMessage, RenderConfig } from "./Protocol";

export class SignalingClient {
  private ws: WebSocket | null = null;
  private myRole: "host" | "worker" | null = null;
  private workers = new Map<string, RtcClient>();
  private hostClient: RtcClient | null = null;

  // Callbacks
  public onStatusChange: ((status: string) => void) | null = null;
  public onWorkerJoined: ((id: string) => void) | null = null;
  public onWorkerLeft: ((id: string) => void) | null = null;
  public onHostConnected: (() => void) | null = null;
  public onWorkerReady: ((id: string) => void) | null = null;

  public onSceneReceived:
    | ((data: ArrayBuffer | string, config: RenderConfig) => void)
    | null = null;
  public onHostHello: (() => void) | null = null;
  public onRenderResult:
    | ((chunks: any[], startFrame: number, workerId: string) => void)
    | null = null;
  public onRenderRequest:
    | ((startFrame: number, frameCount: number, config: RenderConfig) => void)
    | null = null;
  public onWorkerStatus:
    | ((
        id: string,
        hasScene: boolean,
        currentJob?: { start: number; count: number }
      ) => void)
    | null = null;
  public onStopRender: (() => void) | null = null;
  public onSceneLoaded: ((id: string) => void) | null = null;

  constructor() {}

  public connect(role: "host" | "worker") {
    if (this.ws) return;
    this.myRole = role;
    this.onStatusChange?.(`Connecting as ${role.toUpperCase()}...`);

    const token = import.meta.env.VITE_SIGNALING_SECRET || "secretpassword";
    this.ws = new WebSocket(`${Config.signalingServerUrl}?token=${token}`);

    this.ws.onopen = () => {
      console.log("WS Connected");
      this.onStatusChange?.(`Waiting for Peer (${role.toUpperCase()})`);

      if (role === "worker") {
        const sessionId = sessionStorage.getItem("raytracer_session_id");
        const sessionToken = sessionStorage.getItem("raytracer_session_token");
        this.sendSignal({
          type: "register_worker",
          sessionId: sessionId || undefined,
          sessionToken: sessionToken || undefined,
        });
      } else {
        this.sendSignal({ type: "register_host" });
      }
    };

    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as SignalingMessage;
      this.handleMessage(msg);
    };

    this.ws.onclose = () => {
      this.onStatusChange?.("Disconnected");
      this.ws = null;
    };
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    // Also close peer connections
    this.workers.forEach((w) => w.close());
    this.workers.clear();
    if (this.hostClient) {
      this.hostClient.close();
      this.hostClient = null;
    }
    this.onStatusChange?.("Disconnected");
  }

  public getWorkerCount() {
    return this.workers.size;
  }

  public getWorkerIds() {
    return Array.from(this.workers.keys());
  }

  public async sendRenderResult(chunks: any[], startFrame: number) {
    if (this.hostClient) {
      await this.hostClient.sendRenderResult(chunks, startFrame);
    } else {
      throw new Error("No Host Connection");
    }
  }

  private sendSignal(msg: SignalingMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  private async handleMessage(msg: SignalingMessage) {
    if (this.myRole === "host") {
      await this.handleHostMessage(msg);
    } else {
      await this.handleWorkerMessage(msg);
    }
  }

  private async handleHostMessage(msg: SignalingMessage) {
    switch (msg.type) {
      case "worker_joined":
        console.log(`Worker joined: ${msg.workerId}`);
        const setupClient = (id: string) => {
          const client = new RtcClient(id, (m) => this.sendSignal(m));
          this.workers.set(id, client);

          client.onDataChannelOpen = () => {
            console.log(`[Host] Open for ${id}`);
            client.sendData({ type: "HELLO", msg: "Hello from Host!" });
            this.onWorkerJoined?.(id);
          };

          client.onAckReceived = (bytes) => {
            console.log(`Worker ${id} ACK: ${bytes}`);
          };

          client.onRenderResult = (chunks, startFrame) => {
            console.log(
              `Received Render Result from ${id}: ${chunks.length} chunks`
            );
            this.onRenderResult?.(chunks, startFrame, id);
          };

          client.onWorkerReady = () => {
            this.onWorkerReady?.(id);
          };

          client.onWorkerStatus = (hasScene, job) => {
            this.onWorkerStatus?.(id, hasScene, job);
          };

          client.onStopRender = () => {
            this.onStopRender?.();
          };

          client.onSceneLoaded = () => {
            this.onSceneLoaded?.(id);
          };

          client.onConnectionFailure = () => {
            console.warn(`[Host] Connection failed for ${id}. Retrying...`);
            client.close();
            // Wait a bit before retrying to avoid spamming
            setTimeout(() => {
              if (this.workers.has(id)) {
                setupClient(id);
                this.workers.get(id)?.startAsHost();
              }
            }, 2000);
          };

          return client;
        };

        const client = setupClient(msg.workerId);
        await client.startAsHost();
        break;
      case "worker_left":
        console.log(`Worker left: ${msg.workerId}`);
        this.workers.get(msg.workerId)?.close();
        this.workers.delete(msg.workerId);
        this.onWorkerLeft?.(msg.workerId);
        break;
      case "answer":
        if (msg.fromId)
          await this.workers.get(msg.fromId)?.handleAnswer(msg.sdp);
        break;
      case "candidate":
        if (msg.fromId)
          await this.workers.get(msg.fromId)?.handleCandidate(msg.candidate);
        break;
      case "host_exists":
        alert("Host already exists!");
        break;
      case "WORKER_READY":
        if (msg.workerId) this.onWorkerReady?.(msg.workerId);
        break;
    }
  }

  public async sendWorkerReady() {
    // this.sendSignal({ type: "WORKER_READY" }); // Old WS way
    if (this.hostClient) {
      this.hostClient.sendWorkerReady();
    }
  }

  public async sendWorkerStatus(
    hasScene: boolean,
    currentJob?: { start: number; count: number }
  ) {
    if (this.hostClient) {
      this.hostClient.sendWorkerStatus(hasScene, currentJob);
    }
  }

  public async sendSceneLoaded() {
    if (this.hostClient) {
      this.hostClient.sendSceneLoaded();
    }
  }

  private async handleWorkerMessage(msg: SignalingMessage) {
    switch (msg.type) {
      case "session_info":
        console.log(`[Worker] Session Info Received: ${msg.sessionId}`);
        sessionStorage.setItem("raytracer_session_id", msg.sessionId);
        sessionStorage.setItem("raytracer_session_token", msg.sessionToken);
        break;
      case "offer":
        if (msg.fromId) {
          const setupHostClient = (id: string) => {
            if (this.hostClient) this.hostClient.close();
            this.hostClient = new RtcClient(id, (m) => this.sendSignal(m));
            this.onStatusChange?.("Connected to Host!");
            this.onHostConnected?.();

            this.hostClient.onDataChannelOpen = () => {
              this.hostClient?.sendData({
                type: "HELLO",
                msg: "Hello from Worker!",
              });
              this.onHostHello?.();
            };

            this.hostClient.onSceneReceived = (data, config) => {
              this.onSceneReceived?.(data, config);
              // Auto ACK
              const size =
                typeof data === "string" ? data.length : data.byteLength;
              this.hostClient?.sendAck(size);
            };

            this.hostClient.onRenderRequest = (start, count, config) => {
              this.onRenderRequest?.(start, count, config);
            };

            this.hostClient.onStopRender = () => {
              this.onStopRender?.();
            };

            this.hostClient.onSceneLoaded = () => {
              this.onSceneLoaded?.("host"); // Host isn't usually the target of SCENE_LOADED but for symmetry
            };

            this.hostClient.onConnectionFailure = () => {
              console.warn(`[Worker] Connection failed for host ${id}.`);
              // On worker side, we just close and wait for host to send a new offer
              if (this.hostClient) {
                this.hostClient.close();
                this.hostClient = null;
              }
              this.onStatusChange?.("Disconnected from Host (Reconnecting...)");
            };

            return this.hostClient;
          };

          const client = setupHostClient(msg.fromId);
          await client.handleOffer(msg.sdp);
        }
        break;
      case "candidate":
        await this.hostClient?.handleCandidate(msg.candidate);
        break;
    }
  }

  public async broadcastScene(
    fileData: ArrayBuffer | string,
    fileType: "obj" | "glb",
    config: Omit<RenderConfig, "fileType">
  ) {
    const promises = Array.from(this.workers.values()).map((w) =>
      w.sendScene(fileData, fileType, config)
    );
    await Promise.all(promises);
  }

  public async sendSceneToWorker(
    targetId: string,
    fileData: ArrayBuffer | string,
    fileType: "obj" | "glb",
    config: Omit<RenderConfig, "fileType">
  ) {
    const client = this.workers.get(targetId);
    if (!client) {
      console.error(
        `[Host] Cannot send scene to ${targetId}: Client not found.`
      );
      return;
    }

    // Wait for DataChannel to be open
    if (!client.isDataChannelOpen()) {
      console.log(`[Host] DataChannel for ${targetId} not ready, waiting...`);
      await new Promise<void>((resolve) => {
        const onOpen = () => {
          client.dc?.removeEventListener("open", onOpen);
          resolve();
        };
        // If it opens while we wait
        client.dc?.addEventListener("open", onOpen);
        // Fallback timeout
        setTimeout(resolve, 3000);
      });
      // Double check active
      if (!client.isDataChannelOpen()) {
        console.warn(
          `[Host] DataChannel for ${targetId} timed out. Attempting Send anyway...`
        );
      }
    }

    if (client) {
      await client.sendScene(fileData, fileType, config);
    }
  }

  public async sendRenderRequest(
    targetId: string,
    startFrame: number,
    frameCount: number,
    config: RenderConfig
  ) {
    const client = this.workers.get(targetId);
    if (client) {
      await client.sendRenderRequest(startFrame, frameCount, config);
    }
  }

  public sendStopRender(workerId: string) {
    const w = this.workers.get(workerId);
    if (w) {
      w.sendStopRender();
    }
  }

  public sendRenderStart() {
    this.sendSignal({ type: "render_start" });
  }

  public sendRenderStop() {
    this.sendSignal({ type: "render_stop" });
  }
}
