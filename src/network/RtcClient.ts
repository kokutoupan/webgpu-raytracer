// src/network/RtcClient.ts
import type { SignalingMessage, DataChannelMessage } from './Protocol';

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
};

export class RtcClient {
  private pc: RTCPeerConnection;
  private dc: RTCDataChannel | null = null;

  // 相手のID
  public readonly remoteId: string;

  // シグナリング送信用のコールバック
  private sendSignal: (msg: SignalingMessage) => void;

  constructor(remoteId: string, sendSignal: (msg: SignalingMessage) => void) {
    this.remoteId = remoteId;
    this.sendSignal = sendSignal;

    this.pc = new RTCPeerConnection(RTC_CONFIG);

    this.pc.onicecandidate = (ev) => {
      if (ev.candidate) {
        this.sendSignal({
          type: 'candidate',
          candidate: ev.candidate.toJSON(),
          targetId: this.remoteId
        });
      }
    };

    this.pc.onconnectionstatechange = () => {
      console.log(`[RTC ${this.remoteId}] Connection State: ${this.pc.connectionState}`);
    };
  }

  // --- Host として振る舞う場合 ---
  async startAsHost() {
    console.log(`[RTC ${this.remoteId}] Starting as HOST...`);

    this.dc = this.pc.createDataChannel('render-channel');
    this.setupDataChannel();

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    this.sendSignal({
      type: 'offer',
      sdp: offer,
      targetId: this.remoteId
    });
  }

  // --- Worker として振る舞う場合 ---
  async handleOffer(offer: RTCSessionDescriptionInit) {
    console.log(`[RTC ${this.remoteId}] Handling Offer...`);

    this.pc.ondatachannel = (ev) => {
      console.log(`[RTC ${this.remoteId}] DataChannel received!`);
      this.dc = ev.channel;
      this.setupDataChannel();
    };

    await this.pc.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await this.pc.createAnswer();
    await this.pc.setLocalDescription(answer);

    this.sendSignal({
      type: 'answer',
      sdp: answer,
      targetId: this.remoteId
    });
  }

  async handleAnswer(answer: RTCSessionDescriptionInit) {
    await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
  }

  async handleCandidate(candidate: RTCIceCandidateInit) {
    try {
      await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (e) {
      console.error(`[RTC ${this.remoteId}] Error adding ICE candidate`, e);
    }
  }

  private setupDataChannel() {
    if (!this.dc) return;

    this.dc.onopen = () => {
      console.log(`[RTC ${this.remoteId}] DataChannel OPEN!`);
      this.sendData({ type: 'HELLO', msg: `Hello from ${location.hash || 'Browser'}` });
    };

    this.dc.onmessage = (ev) => {
      try {
        // ここは実行時に値として使うので import type は不要（JSON.parseの結果）
        const msg = JSON.parse(ev.data) as DataChannelMessage;
        console.log(`[RTC ${this.remoteId}] RECV:`, msg);
      } catch (e) {
        console.warn('Unknown message format:', ev.data);
      }
    };

    this.dc.onerror = (err) => console.error(`[RTC ${this.remoteId}] DC Error:`, err);
  }

  public sendData(msg: DataChannelMessage) {
    if (this.dc?.readyState === 'open') {
      this.dc.send(JSON.stringify(msg));
    } else {
      console.warn(`[RTC ${this.remoteId}] Cannot send, DC not open.`);
    }
  }
}
