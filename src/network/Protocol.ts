export type SignalingMessage =
  | { type: "register_host" }
  | { type: "host_exists" }
  | { type: "register_worker" }
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
    };

// レンダリング設定（Res + Recの内容）
export type RenderConfig = {
  width: number;
  height: number;
  fps: number;
  duration: number;
  spp: number; // RecのSPP
  batch: number; // RecのBatch
  fileType: "obj" | "glb"; // ファイル形式
  anim: number; // Animation Index
};

// DataChannelで送るデータ
export type DataChannelMessage =
  | { type: "HELLO"; msg: string }
  // 設定とファイルサイズの通知
  | { type: "SCENE_INIT"; config: RenderConfig; totalBytes: number }
  // 受信完了通知
  | { type: "SCENE_ACK"; receivedBytes: number };
