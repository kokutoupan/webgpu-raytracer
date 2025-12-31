export type SignalingMessage =
  | { type: "register_host" }
  | { type: "render_start" }
  | { type: "render_stop" }
  | { type: "host_exists" }
  | {
      type: "register_worker";
      sessionId?: string;
      sessionToken?: string;
    }
  | {
      type: "session_info";
      sessionId: string;
      sessionToken: string;
    }
  | { type: "worker_joined"; workerId: string }
  | { type: "worker_left"; workerId: string }
  | {
      type: "offer";
      sdp: RTCSessionDescriptionInit;
      targetId: string;
      fromId?: string;
    }
  | {
      type: "answer";
      sdp: RTCSessionDescriptionInit;
      targetId: string;
      fromId?: string;
    }
  | {
      type: "candidate";
      candidate: RTCIceCandidateInit;
      targetId: string;
      fromId?: string;
    }
  | { type: "WORKER_READY"; workerId?: string }; // Sent by worker when scene is fully loaded

// レンダリング設定（Res + Recの内容）
export type RenderConfig = {
  width: number;
  height: number;
  fps: number;
  duration: number;
  spp: number; // RecのSPP
  batch: number; // RecのBatch
  jobBatch?: number; // Distributed Job Size
  fileType: "obj" | "glb"; // ファイル形式
  anim: number; // Animation Index
  sceneName?: string; // For procedural scenes (e.g. "cornell_box")
  maxDepth: number; // Shader Max Depth
  shaderSpp: number; // Samples per dispatch
};

// DataChannelで送るデータ
export interface SerializedChunk {
  type: "key" | "delta";
  timestamp: number;
  duration: number;
  data: ArrayBuffer;
  // EncodedVideoChunkMetadata usually has decoderConfig
  decoderConfig?: VideoDecoderConfig | null;
}

export type DataChannelMessage =
  | { type: "HELLO"; msg: string }
  // 設定とファイルサイズの通知
  | {
      type: "SCENE_INIT";
      totalBytes: number;
      config: RenderConfig;
    }
  // 受信完了通知
  | {
      type: "SCENE_ACK";
      receivedBytes: number;
    }
  // レンダリング依頼 (Host -> Worker)
  | {
      type: "RENDER_REQUEST";
      startFrame: number;
      frameCount: number;
      config: RenderConfig;
    }
  // レンダリング結果 (Worker -> Host)
  | {
      type: "RENDER_RESULT";
      startFrame: number;
      totalBytes: number;
      chunksMeta: {
        type: "key" | "delta";
        timestamp: number;
        duration: number;
        size: number;
        decoderConfig?: VideoDecoderConfig | null;
      }[];
    }
  | {
      type: "WORKER_STATUS";
      hasScene: boolean;
      currentJob?: { start: number; count: number };
    }
  | { type: "STOP_RENDER" }
  | { type: "WORKER_READY" }
  | { type: "SCENE_LOADED" };
