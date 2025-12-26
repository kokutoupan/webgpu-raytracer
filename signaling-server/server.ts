import { WebSocket, WebSocketServer } from "ws";
import dotenv from "dotenv";
import path from "path";
import { randomUUID, timingSafeEqual } from "crypto";

// 環境変数の読み込み
dotenv.config({ path: path.resolve(__dirname, "../.env") });

// ポート番号
const PORT = Number(process.env.VITE_SIGNALING_SERVER_PORT) || 8080;
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

// --- 状態管理 ---

let hostSocket: ExtWebSocket | null = null;
const workers = new Map<string, ExtWebSocket>();
// Persistent sessions: id -> token
const sessions = new Map<string, string>();

// --- メインロジック ---

const MAX_PAYLOAD = 64 * 1024; // 64KB
const SECRET = process.env.VITE_SIGNALING_SECRET || "secretpassword";

wss.on("connection", (ws: ExtWebSocket, req) => {
  // Authentication
  // 1. URL解析
  const url = new URL(req.url || "", `http://${req.headers.host}`);
  const token = url.searchParams.get("token");

  // 2. SECRETがない場合はサーバー設定ミスとして落とす（Fail Closed）
  if (!SECRET) {
    console.error("Critical Error: SECRET is not configured on server.");
    ws.close(1011, "Server Configuration Error");
    return;
  }

  // 3. トークンがない、または長さが違う場合は即切断
  if (!token || token.length !== SECRET.length) {
    console.log(
      `Connection rejected: Missing or invalid length token from ${req.socket.remoteAddress}`
    );
    ws.close(1008, "Invalid Token");
    return;
  }

  // 4. タイミング攻撃に強い比較（Constant Time Comparison）
  // Bufferに変換して crypto.timingSafeEqual で比較する
  const sourceBuffer = Buffer.from(token);
  const targetBuffer = Buffer.from(SECRET);

  // 長さが違うとtimingSafeEqualはエラーになるので上記3で弾いておく必要がある
  const isValid = timingSafeEqual(sourceBuffer, targetBuffer);

  if (!isValid) {
    console.log(
      `Connection rejected: Invalid token from ${req.socket.remoteAddress}`
    );
    ws.close(1008, "Invalid Token");
    return;
  }

  // 簡易ID生成 (ランダム文字列)
  ws.id = randomUUID();

  console.log(`New connection: ${ws.id}`);

  ws.on("message", (message: Buffer | string) => {
    // Size Limit
    if (message.length > MAX_PAYLOAD) {
      console.warn(`Message too large from ${ws.id}: ${message.length} bytes`);
      ws.close(1009, "Message too large");
      return;
    }

    let data: any;
    try {
      data = JSON.parse(message.toString());
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
        let finalId = ws.id;
        // Session Resumption
        if (data.sessionId && data.sessionToken) {
          const storedToken = sessions.get(data.sessionId);
          if (storedToken && storedToken === data.sessionToken) {
            console.log(`Worker session resumed: ${data.sessionId}`);
            finalId = data.sessionId;

            // If there's an old socket, close it
            const oldWs = workers.get(finalId);
            if (oldWs && oldWs !== ws && oldWs.readyState === WebSocket.OPEN) {
              oldWs.close();
            }
          } else {
            console.warn(
              `Invalid session resumption attempt: ${data.sessionId}`
            );
          }
        }

        // Assign/Update ID
        ws.id = finalId;
        console.log(`Worker registered: ${ws.id}`);
        workers.set(ws.id, ws);
        ws.role = "worker";

        // Generate new session token if it's a new session or we want to refresh
        if (!sessions.has(ws.id)) {
          const sessionToken = randomUUID();
          sessions.set(ws.id, sessionToken);
          sendTo(ws, { type: "session_info", sessionId: ws.id, sessionToken });
        } else {
          // Send existing session info to acknowledge resumption
          sendTo(ws, {
            type: "session_info",
            sessionId: ws.id,
            sessionToken: sessions.get(ws.id)!,
          });
        }

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
