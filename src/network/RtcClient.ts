// src/network/RtcClient.ts
import type {
  SignalingMessage,
  DataChannelMessage,
  RenderConfig,
} from "./Protocol";

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

export class RtcClient {
  // ... (既存のプロパティ等はそのまま)
  private pc: RTCPeerConnection;
  private dc: RTCDataChannel | null = null;
  public readonly remoteId: string;
  private sendSignal: (msg: SignalingMessage) => void;

  // 受信バッファ
  private receiveBuffer: Uint8Array = new Uint8Array(0);
  private receivedBytes = 0;
  private sceneMeta: { config: RenderConfig; totalBytes: number } | null = null;
  private resultMeta: {
    startFrame: number;
    totalBytes: number;
    chunksMeta: any[];
  } | null = null;

  // コールバック
  public onSceneReceived:
    | ((data: ArrayBuffer | string, config: RenderConfig) => void)
    | null = null;
  public onRenderRequest:
    | ((startFrame: number, frameCount: number, config: RenderConfig) => void)
    | null = null;
  public onRenderResult: ((chunks: any[], startFrame: number) => void) | null =
    null;
  public onDataChannelOpen: (() => void) | null = null;
  public onAckReceived: ((receivedBytes: number) => void) | null = null;
  public onWorkerReady: (() => void) | null = null;

  constructor(remoteId: string, sendSignal: (msg: SignalingMessage) => void) {
    this.remoteId = remoteId;
    this.sendSignal = sendSignal;
    this.pc = new RTCPeerConnection(RTC_CONFIG);
    // ... (ICE Candidate等の処理は既存のまま) ...
    this.pc.onicecandidate = (ev) => {
      if (ev.candidate)
        this.sendSignal({
          type: "candidate",
          candidate: ev.candidate.toJSON(),
          targetId: this.remoteId,
        });
    };
  }

  // ... (startAsHost, handleOffer 等の接続処理は既存のまま) ...
  async startAsHost() {
    this.dc = this.pc.createDataChannel("render-channel");
    this.setupDataChannel();
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.sendSignal({ type: "offer", sdp: offer, targetId: this.remoteId });
  }

  async handleOffer(offer: RTCSessionDescriptionInit) {
    this.pc.ondatachannel = (ev) => {
      this.dc = ev.channel;
      this.setupDataChannel();
    };
    await this.pc.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await this.pc.createAnswer();
    await this.pc.setLocalDescription(answer);
    this.sendSignal({ type: "answer", sdp: answer, targetId: this.remoteId });
  }

  async handleAnswer(answer: RTCSessionDescriptionInit) {
    await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
  }
  async handleCandidate(candidate: RTCIceCandidateInit) {
    await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
  }

  // ============================================
  //  送信ロジック (Host)
  // ============================================
  public async sendScene(
    fileData: ArrayBuffer | string,
    fileType: "obj" | "glb",
    config: Omit<RenderConfig, "fileType">
  ) {
    if (!this.dc || this.dc.readyState !== "open") return;

    let buffer: Uint8Array;
    if (typeof fileData === "string") {
      buffer = new TextEncoder().encode(fileData);
    } else {
      buffer = new Uint8Array(fileData);
    }

    // 1. メタデータ送信
    const metaMsg: DataChannelMessage = {
      type: "SCENE_INIT",
      totalBytes: buffer.byteLength,
      config: { ...config, fileType },
    };
    this.sendData(metaMsg);

    // 2. チャンク送信
    await this.sendBinaryChunks(buffer);
  }

  public async sendRenderResult(chunks: any[], startFrame: number) {
    if (!this.dc || this.dc.readyState !== "open") return;

    // Calculate total size and prepare metadata
    let totalBytes = 0;
    const chunksMeta = chunks.map((c) => {
      const size = c.data.byteLength;
      totalBytes += size;
      return {
        type: c.type,
        timestamp: c.timestamp,
        duration: c.duration,
        size,
        decoderConfig: c.decoderConfig,
      };
    });

    console.log(
      `[RTC] Sending Render Result: ${totalBytes} bytes, ${chunks.length} chunks`
    );

    this.sendData({
      type: "RENDER_RESULT",
      startFrame,
      totalBytes,
      chunksMeta,
    });

    // Combine buffers
    const combinedBuffer = new Uint8Array(totalBytes);
    let offset = 0;
    for (const c of chunks) {
      combinedBuffer.set(new Uint8Array(c.data), offset);
      offset += c.data.byteLength;
    }

    await this.sendBinaryChunks(combinedBuffer);
  }

  private async sendBinaryChunks(buffer: Uint8Array) {
    const CHUNK_SIZE = 16 * 1024;
    let offset = 0;

    const waitBuffer = () =>
      new Promise<void>((resolve) => {
        const i = setInterval(() => {
          if (!this.dc || this.dc.bufferedAmount < 64 * 1024) {
            clearInterval(i);
            resolve();
          }
        }, 5);
      });

    while (offset < buffer.byteLength) {
      if (this.dc && this.dc.bufferedAmount > 256 * 1024) await waitBuffer();

      const end = Math.min(offset + CHUNK_SIZE, buffer.byteLength);
      if (this.dc) {
        try {
          this.dc.send(buffer.subarray(offset, end) as any);
        } catch (e) {
          /* ignore closed */
        }
      }
      offset = end;

      if (offset % (CHUNK_SIZE * 5) === 0)
        await new Promise((r) => setTimeout(r, 0));
    }
    console.log(`[RTC] Transfer Complete`);
  }

  // ============================================
  //  受信ロジック (Worker)
  // ============================================
  private setupDataChannel() {
    if (!this.dc) return;
    this.dc.binaryType = "arraybuffer";

    this.dc.onopen = () => {
      console.log(`[RTC] DataChannel Open`);
      if (this.onDataChannelOpen) this.onDataChannelOpen();
    };

    this.dc.onmessage = (ev) => {
      const data = ev.data;
      if (typeof data === "string") {
        try {
          const msg = JSON.parse(data) as DataChannelMessage;
          this.handleControlMessage(msg);
        } catch (e) {}
      } else if (data instanceof ArrayBuffer) {
        this.handleBinaryChunk(data);
      }
    };
  }

  private handleControlMessage(msg: DataChannelMessage) {
    if (msg.type === "SCENE_INIT") {
      console.log(
        `[RTC] Receiving Scene: ${msg.config.fileType}, ${msg.totalBytes} bytes`
      );
      this.sceneMeta = { config: msg.config, totalBytes: msg.totalBytes };
      this.receiveBuffer = new Uint8Array(msg.totalBytes);
      this.receivedBytes = 0;
    } else if (msg.type === "SCENE_ACK") {
      console.log(`[RTC] Scene ACK: ${msg.receivedBytes} bytes`);
      if (this.onAckReceived) this.onAckReceived(msg.receivedBytes);
    } else if (msg.type === "RENDER_REQUEST") {
      console.log(
        `[RTC] Render Request: Frame ${msg.startFrame}, Count ${msg.frameCount}`
      );
      this.onRenderRequest?.(msg.startFrame, msg.frameCount, msg.config);
    } else if (msg.type === "RENDER_RESULT") {
      console.log(`[RTC] Receiving Render Result: ${msg.totalBytes} bytes`);
      this.resultMeta = {
        startFrame: msg.startFrame,
        totalBytes: msg.totalBytes,
        chunksMeta: msg.chunksMeta,
      };
      this.receiveBuffer = new Uint8Array(msg.totalBytes);
      this.receivedBytes = 0;
    } else if (msg.type === "WORKER_READY") {
      console.log(`[RTC] Worker Ready Signal Received`);
      this.onWorkerReady?.();
    }
  }

  private handleBinaryChunk(chunk: ArrayBuffer) {
    const data = new Uint8Array(chunk);
    this.receiveBuffer.set(data, this.receivedBytes);
    this.receivedBytes += data.byteLength;

    if (this.sceneMeta) {
      if (this.receivedBytes >= this.sceneMeta.totalBytes) {
        console.log(`[RTC] Scene Download Complete!`);
        let resultData: ArrayBuffer | string;
        if (this.sceneMeta.config.fileType === "obj") {
          resultData = new TextDecoder().decode(this.receiveBuffer);
        } else {
          resultData = this.receiveBuffer.buffer as ArrayBuffer;
        }

        this.onSceneReceived?.(resultData, this.sceneMeta.config);
        this.sceneMeta = null;
      }
    } else if (this.resultMeta) {
      if (this.receivedBytes >= this.resultMeta.totalBytes) {
        console.log(`[RTC] Render Result Complete!`);

        // Reconstruct Chunks
        const chunks: any[] = [];
        let offset = 0;
        for (const meta of this.resultMeta.chunksMeta) {
          const data = this.receiveBuffer.slice(offset, offset + meta.size);
          chunks.push({
            type: meta.type,
            timestamp: meta.timestamp,
            duration: meta.duration,
            data: data.buffer,
            decoderConfig: meta.decoderConfig,
          });
          offset += meta.size;
        }

        this.onRenderResult?.(chunks, this.resultMeta.startFrame);
        this.resultMeta = null;
      }
    }
  }

  public sendData(msg: DataChannelMessage) {
    if (this.dc?.readyState === "open") this.dc.send(JSON.stringify(msg));
  }

  public sendAck(receivedBytes: number) {
    this.sendData({ type: "SCENE_ACK", receivedBytes });
  }

  public sendRenderRequest(
    startFrame: number,
    frameCount: number,
    config: RenderConfig
  ) {
    const msg: DataChannelMessage = {
      type: "RENDER_REQUEST",
      startFrame,
      frameCount,
      config,
    };
    this.sendData(msg);
  }

  public sendWorkerReady() {
    this.sendData({ type: "WORKER_READY" });
  }

  public close() {
    if (this.dc) {
      this.dc.close();
      this.dc = null;
    }
    if (this.pc) {
      this.pc.close();
    }
    console.log(`[RTC] Connection closed: ${this.remoteId}`);
  }
}
