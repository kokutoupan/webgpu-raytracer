import { WebSocket, WebSocketServer } from "ws";

// ポート番号
const PORT = 8080;

// サーバーの起動
const wss = new WebSocketServer({ port: PORT });
console.log(`Signaling Server running on port ${PORT}`);

// --- 型定義 ---

// 拡張したWebSocket型（IDとRoleを持つ）
interface ExtWebSocket extends WebSocket {
  id: string;
  role?: "host" | "worker";
}

// クライアントから送られてくるメッセージの型
type SignalingMessage =
  | { type: "register_host" }
  | { type: "register_worker" }
  | { type: "offer"; sdp: any; targetId: string; fromId?: string }
  | { type: "answer"; sdp: any; targetId: string; fromId?: string }
  | { type: "candidate"; candidate: any; targetId: string; fromId?: string };

// --- 状態管理 ---

let hostSocket: ExtWebSocket | null = null;
const workers = new Map<string, ExtWebSocket>();

// --- メインロジック ---

wss.on("connection", (ws: ExtWebSocket) => {
  // 簡易ID生成 (ランダム文字列)
  ws.id = Math.random().toString(36).substring(2, 9);

  console.log(`New connection: ${ws.id}`);

  ws.on("message", (message: string) => {
    let data: SignalingMessage;
    try {
      data = JSON.parse(message.toString()) as SignalingMessage;
    } catch (e) {
      console.error("Invalid JSON:", message);
      return;
    }

    switch (data.type) {
      case "register_host": {
        console.log(`Host registered: ${ws.id}`);
        // Allow overwriting host for development ease / reconnection
        if (hostSocket != null && hostSocket.readyState === WebSocket.OPEN) {
          console.log("Replacing existing host");
          hostSocket.close();
        }

        hostSocket = ws;
        ws.role = "host";

        // Notify Host of existing workers
        console.log(`Notifying Host of ${workers.size} existing workers.`);
        for (const [workerId, workerWs] of workers) {
          if (workerWs.readyState === WebSocket.OPEN) {
            sendTo(ws, {
              type: "worker_joined",
              workerId: workerId,
            });
          }
        }
        break;
      }

      case "register_worker": {
        console.log(`Worker registered: ${ws.id}`);
        workers.set(ws.id, ws);
        ws.role = "worker";

        // Hostに通知
        if (hostSocket && hostSocket.readyState === WebSocket.OPEN) {
          sendTo(hostSocket, {
            type: "worker_joined",
            workerId: ws.id,
          });
        }
        break;
      }

      // WebRTC シグナリング (転送処理)
      case "offer":
      case "answer":
      case "candidate": {
        const targetId = data.targetId;
        let targetWs: ExtWebSocket | undefined;

        // 【修正】 'HOST' という指定、または HostのIDと一致する場合に Host を対象にする
        if (targetId === "HOST" || (hostSocket && targetId === hostSocket.id)) {
          targetWs = hostSocket || undefined;
        } else {
          targetWs = workers.get(targetId);
        }

        if (targetWs && targetWs.readyState === WebSocket.OPEN) {
          // 送信元IDを付与して転送
          data.fromId = ws.id;
          sendTo(targetWs, data);
        } else {
          console.warn(`Target ${targetId} not found or closed.`);
        }
        break;
      }
    }
  });

  ws.on("close", () => {
    if (ws === hostSocket) {
      console.log("Host disconnected");
      hostSocket = null;
    } else if (workers.has(ws.id)) {
      console.log(`Worker disconnected: ${ws.id}`);
      workers.delete(ws.id);

      // Hostに通知
      if (hostSocket && hostSocket.readyState === WebSocket.OPEN) {
        sendTo(hostSocket, {
          type: "worker_left",
          workerId: ws.id,
        });
      }
    }
  });
});

// ヘルパー関数: JSON送信
function sendTo(ws: WebSocket, msg: any) {
  ws.send(JSON.stringify(msg));
}
