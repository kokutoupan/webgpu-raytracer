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

  constructor() {}

  public connect(role: "host" | "worker") {
    if (this.ws) return;
    this.myRole = role;
    this.onStatusChange?.(`Connecting as ${role.toUpperCase()}...`);

    this.ws = new WebSocket(Config.signalingServerUrl);

    this.ws.onopen = () => {
      console.log("WS Connected");
      this.onStatusChange?.(`Waiting for Peer (${role.toUpperCase()})`);
      this.sendSignal({
        type: role === "host" ? "register_host" : "register_worker",
      });
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
      // Blob to ArrayBuffer
      // const buffer = await blob.arrayBuffer(); // No longer needed
      await this.hostClient.sendRenderResult(chunks, startFrame);
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
        const client = new RtcClient(msg.workerId, (m) => this.sendSignal(m));
        this.workers.set(msg.workerId, client);

        client.onDataChannelOpen = () => {
          console.log(`[Host] Open for ${msg.workerId}`);
          // Initial Hello
          client.sendData({ type: "HELLO", msg: "Hello from Host!" });
          this.onWorkerJoined?.(msg.workerId);
        };

        client.onAckReceived = (bytes) => {
          console.log(`Worker ${msg.workerId} ACK: ${bytes}`);
        };

        client.onRenderResult = (chunks, startFrame) => {
          console.log(
            `Received Render Result from ${msg.workerId}: ${chunks.length} chunks`
          );
          this.onRenderResult?.(chunks, startFrame, msg.workerId);
        };

        client.onWorkerReady = () => {
          this.onWorkerReady?.(msg.workerId);
        };

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

  private async handleWorkerMessage(msg: SignalingMessage) {
    switch (msg.type) {
      case "offer":
        if (msg.fromId) {
          this.hostClient = new RtcClient(msg.fromId, (m) =>
            this.sendSignal(m)
          );
          await this.hostClient.handleOffer(msg.sdp);
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

  // ... (broadcastRenderRequest can stay or be removed if unused)
}
