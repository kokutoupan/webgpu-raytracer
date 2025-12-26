import { WebSocket, WebSocketServer } from "ws";
import dotenv from "dotenv";
import path from "path";
import http from "http";
import fs from "fs";
import { randomUUID, timingSafeEqual } from "crypto";

// 環境変数の読み込み
dotenv.config({ path: path.resolve(__dirname, "../.env") });

// ポート番号
const PORT = Number(process.env.VITE_SIGNALING_SERVER_PORT) || 8080;
const ADMIN_USERNAME = process.env.ADMIN_USERNAME || "admin";
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "admin";

// --- 日志キャプチャ ---
const LOG_LIMIT = 100;
const logs: { time: string; level: string; msg: string }[] = [];

function addLog(level: string, ...args: any[]) {
  const msg = args
    .map((a) => (typeof a === "object" ? JSON.stringify(a) : String(a)))
    .join(" ");
  const time = new Date().toISOString();
  logs.push({ time, level, msg });
  if (logs.length > LOG_LIMIT) logs.shift();

  // 標準出力にも出す (再帰防止のため process.stdout/stderr を直接使用)
  if (level === "ERROR") process.stderr.write(`[${time}] ${msg}\n`);
  else process.stdout.write(`[${time}] ${msg}\n`);
}

// console.logなどをラップ（オプションだが管理コンソール用には便利）
console.log = (...args) => {
  addLog("INFO", ...args);
};
console.error = (...args) => {
  addLog("ERROR", ...args);
};

// --- HTTP サーバー (Basic Auth & Admin Page) ---

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "", `http://${req.headers.host}`);

  // Basic Auth Check for /admin
  if (url.pathname.startsWith("/admin")) {
    const auth = req.headers.authorization;
    if (!auth) {
      res.setHeader("WWW-Authenticate", 'Basic realm="Admin Console"');
      res.writeHead(401);
      res.end("Authentication required");
      return;
    }

    const [username, password] = Buffer.from(auth.split(" ")[1], "base64")
      .toString()
      .split(":");
    if (username !== ADMIN_USERNAME || password !== ADMIN_PASSWORD) {
      res.setHeader("WWW-Authenticate", 'Basic realm="Admin Console"');
      res.writeHead(401);
      res.end("Invalid credentials");
      return;
    }

    // Routing
    if (url.pathname === "/admin" || url.pathname === "/admin/") {
      const htmlPath = path.resolve(__dirname, "admin.html");
      if (fs.existsSync(htmlPath)) {
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(fs.readFileSync(htmlPath));
      } else {
        res.writeHead(404);
        res.end("admin.html not found. Please create it.");
      }
      return;
    }

    if (url.pathname === "/admin/api/status") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          host: hostSocket
            ? { id: hostSocket.id, address: hostSocket.remoteAddress }
            : null,
          workers: Array.from(workers.entries()).map(([id, ws]) => ({
            id,
            address: ws.remoteAddress,
          })),
          logs: logs,
        })
      );
      return;
    }

    if (url.pathname === "/admin/api/kick-host" && req.method === "POST") {
      if (hostSocket) {
        console.log(`[Admin] Kicking host: ${hostSocket.id}`);
        hostSocket.close(1001, "Kicked by Admin");
        hostSocket = null;
        res.writeHead(200);
        res.end("Host kicked");
      } else {
        res.writeHead(400);
        res.end("No host connected");
      }
      return;
    }
  }

  res.writeHead(404);
  res.end("Not Found");
});

// WebSocket Server attach to HTTP server
const wss = new WebSocketServer({ server });

server.listen(PORT, () => {
  console.log(`Signaling Server + Admin Console running on port ${PORT}`);
});

// --- 型定義 ---

// 拡張したWebSocket型（IDとRoleを持つ）
interface ExtWebSocket extends WebSocket {
  id: string;
  role?: "host" | "worker";
  remoteAddress?: string;
}

// クライアントから送られてくるメッセージの型
type SignalingMessage =
  | { type: "register_host" }
  | { type: "register_worker" }
  | { type: "render_start" }
  | { type: "render_stop" }
  | { type: "offer"; sdp: any; targetId: string; fromId?: string }
  | { type: "answer"; sdp: any; targetId: string; fromId?: string }
  | { type: "candidate"; candidate: any; targetId: string; fromId?: string };

// --- 状態管理 ---

let hostSocket: ExtWebSocket | null = null;
const workers = new Map<string, ExtWebSocket>();
// Persistent sessions: id -> token
const sessions = new Map<string, string>();

// --- メインロジック ---

const MAX_PAYLOAD = 64 * 1024; // 64KB
const SECRET = process.env.VITE_SIGNALING_SECRET || "secretpassword";

wss.on("connection", (ws: ExtWebSocket, req) => {
  ws.remoteAddress = req.socket.remoteAddress;

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
  const sourceBuffer = Buffer.from(token);
  const targetBuffer = Buffer.from(SECRET);

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

  console.log(`New connection: ${ws.id} from ${ws.remoteAddress}`);

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
        // First-come, first-served: If a host already exists, reject the new one.
        if (hostSocket != null && hostSocket.readyState === WebSocket.OPEN) {
          console.warn(
            `[Refused] Host registration from ${ws.id} refused. Host ${hostSocket.id} is already active.`
          );
          sendTo(ws, { type: "host_exists" });
          return;
        }

        console.log(`Host registered: ${ws.id}`);
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

      case "render_start": {
        console.log(`[Host] Distributed Render Started.`);
        break;
      }

      case "render_stop": {
        console.log(`[Host] Distributed Render Finished.`);
        break;
      }

      // WebRTC シグナリング (転送処理)
      case "offer":
      case "answer":
      case "candidate": {
        const targetId = data.targetId;
        let targetWs: ExtWebSocket | undefined;

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
