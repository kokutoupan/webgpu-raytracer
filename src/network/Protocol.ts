export type SignalingMessage =
  | { type: 'register_host' }
  | { type: 'register_worker' }
  | { type: 'worker_joined', workerId: string }
  | { type: 'worker_left', workerId: string }
  | { type: 'offer', sdp: RTCSessionDescriptionInit, targetId: string, fromId?: string }
  | { type: 'answer', sdp: RTCSessionDescriptionInit, targetId: string, fromId?: string }
  | { type: 'candidate', candidate: RTCIceCandidateInit, targetId: string, fromId?: string };

// DataChannelで送るデータ
export type DataChannelMessage =
  | { type: 'HELLO', msg: string }
  | { type: 'SCENE_CHUNK', data: Uint8Array }; // 今後のため
