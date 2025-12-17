export const Config = {
  // Rendering Defaults
  defaultWidth: 720,
  defaultHeight: 480,
  defaultDepth: 10,
  defaultSPP: 1,
  defaultUpdateInterval: 4,

  // WebRTC / Network
  signalingServerUrl:
    import.meta.env.VITE_SIGNALING_SERVER_URL || "ws://localhost:8080",
  rtcConfig: {
    iceServers: import.meta.env.VITE_ICE_SERVERS
      ? JSON.parse(import.meta.env.VITE_ICE_SERVERS)
      : [{ urls: "stun:stun.l.google.com:19302" }],
  } as RTCConfiguration,

  // DOM Layout IDs
  ids: {
    canvas: "gpu-canvas",
    renderBtn: "render-btn",
    sceneSelect: "scene-select",
    resWidth: "res-width",
    resHeight: "res-height",
    objFile: "obj-file",
    maxDepth: "max-depth",
    sppFrame: "spp-frame",
    recompileBtn: "recompile-btn",
    updateInterval: "update-interval",
    animSelect: "anim-select",

    // Recorder UI
    recordBtn: "record-btn",
    recFps: "rec-fps",
    recDuration: "rec-duration",
    recSpp: "rec-spp",
    recBatch: "rec-batch",

    // Network UI
    btnHost: "btn-host",
    btnWorker: "btn-worker",
    btnSendScene: "btn-send-scene",
    statusDiv: "status",
  },
};
