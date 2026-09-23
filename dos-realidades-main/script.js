// ============================================================================
// Dos realidades — Ejercicio 02 (DPPI 2026)
//
// Una misma cámara alimenta a dos sistemas de visión artificial independientes:
//
//   SISTEMA A (visión morfológica/gestual): MediaPipe Hand Landmarker.
//     Pregunta: ¿dónde está la mano y qué gesto realiza el cuerpo?
//     Representación: un lienzo donde la punta del dedo índice traza líneas
//     en el aire al apuntar, flotando sobre una constelación sutil de la mano.
//
//   SISTEMA B (visión cromática/espectral): segmentación y reconocimiento de color.
//     Pregunta: ¿dónde aparece un color específico en la escena física?
//     Representación: un campo de luminancia que resalta en tiempo real los
//     objetos que coinciden con el color seleccionado (vía retícula central,
//     clic sobre la imagen o paleta de presets).
//
// CONEXIÓN ENTRE AMBAS REALIDADES:
//   El color aislado por el Sistema B (la luz del objeto) se convierte
//   en la tinta con la que el Sistema A (el gesto corporal) dibuja en el espacio.
// ============================================================================

// ---------------------------------------------------------------------------
// Configuración general
// ---------------------------------------------------------------------------

const VISION_BUNDLE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Resolución de muestreo para procesamiento de color del Sistema B
const SAMPLE_COLS = 160;
const SAMPLE_ROWS = 120;

// Topología de la mano (21 landmarks de MediaPipe Hands)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // Pulgar
  [0, 5], [5, 6], [6, 7], [7, 8],        // Índice
  [5, 9], [9, 10], [10, 11], [11, 12],   // Medio
  [9, 13], [13, 14], [14, 15], [15, 16], // Anular
  [13, 17], [17, 18], [18, 19], [19, 20],// Meñique
  [0, 17],                               // Base de la palma
];

// Colores sutiles para la visualización anatómica de la mano
const FINGER_COLORS = {
  0: "#ffffff",
  1: "#ff8a5c", 2: "#ff8a5c", 3: "#ff8a5c", 4: "#ff8a5c",
  5: "#ffd166", 6: "#ffd166", 7: "#ffd166", 8: "#ffd166",
  9: "#5ce1ff", 10: "#5ce1ff", 11: "#5ce1ff", 12: "#5ce1ff",
  13: "#a78bfa", 14: "#a78bfa", 15: "#a78bfa", 16: "#a78bfa",
  17: "#ff5c8a", 18: "#ff5c8a", 19: "#ff5c8a", 20: "#ff5c8a",
};

// ---------------------------------------------------------------------------
// Referencias DOM
// ---------------------------------------------------------------------------

const video = document.getElementById("video");
const canvasA = document.getElementById("canvasA");
const ctxA = canvasA.getContext("2d");
const canvasB = document.getElementById("canvasB");
const ctxB = canvasB.getContext("2d");
const hiddenSample = document.getElementById("hiddenSample");
const ctxHidden = hiddenSample.getContext("2d", { willReadFrequently: true });

const startBtn = document.getElementById("startBtn");
const cameraBtnWrapper = document.getElementById("cameraBtnWrapper");
const clearBtn = document.getElementById("clearBtn");
const statusMsg = document.getElementById("statusMsg");
const idleHintA = document.getElementById("idleHintA");
const gestureBadge = document.getElementById("gestureBadge");
const statA = document.getElementById("statA");
const statB = document.getElementById("statB");
const detectionBar = document.getElementById("detectionBar");
const detectionPercent = document.getElementById("detectionPercent");
const detectionBarTrack = detectionBar.parentElement;
const detectionChart = detectionBarTrack.parentElement;

// Controles de color
const colorPreview = document.getElementById("colorPreview");
const colorHex = document.getElementById("colorHex");
const sampleColorBtn = document.getElementById("sampleColorBtn");
const presetChips = document.querySelectorAll(".chip");

// ---------------------------------------------------------------------------
// Estado
// ---------------------------------------------------------------------------

let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

// Estado de dibujo (Sistema A)
let paths = [];
let currentPath = null;
let isDrawing = false;
let smoothIndexPos = null;
let lastIndexSeenAt = 0;
let lastPointingAt = 0;

const TIP_TRACKING_GRACE_MS = 900;
const GESTURE_TRACKING_GRACE_MS = 220;

// Estado de color (Sistema B)
let targetRgb = { r: 229, g: 57, b: 53 };
let targetHsv = rgbToHsv(229, 57, 53);
let targetHex = "#E53935";
let rippleFeedback = null;

// Configurar dimensiones de muestreo
hiddenSample.width = SAMPLE_COLS;
hiddenSample.height = SAMPLE_ROWS;
// ---------------------------------------------------------------------------
// Inicialización y eventos
// ---------------------------------------------------------------------------

startBtn.addEventListener("click", () => {
  start();
});
clearBtn.addEventListener("click", clearDrawing);
sampleColorBtn.addEventListener("click", sampleCenterColor);

// Selección de color mediante presets
presetChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const hex = chip.getAttribute("data-color");
    setTargetColorFromHex(hex);
    presetChips.forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
  });
});

// Selección de color haciendo clic directo en el canvas B
canvasB.addEventListener("click", (event) => {
  if (!running) return;
  sampleColorFromCanvasClick(event);
});

function clearDrawing() {
  paths = [];
  currentPath = null;
  ctxA.clearRect(0, 0, canvasA.width, canvasA.height);
}

// No se intenta arrancar automáticamente en contextos inseguros.
// La cámara solo debe solicitarse cuando el usuario hace click desde localhost/https.
window.addEventListener("DOMContentLoaded", () => {
  if (window.location.protocol === "file:") {
    setStatus("La cámara no funciona si abres este archivo directamente. Usa http://localhost:8000 ejecutando iniciar_servidor.bat.");
    startBtn.disabled = false;
    return;
  }

  if (!window.isSecureContext && !(window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "[::1]")) {
    setStatus("Abre la página desde http://localhost:8000 para usar la cámara.");
    startBtn.disabled = false;
    return;
  }

  setStatus("Solicitando acceso a la cámara…");
  start(true);
});

// ---------------------------------------------------------------------------
// Arranque de cámara y modelo
// ---------------------------------------------------------------------------

async function start(isAutoAttempt = false) {
  if (running) return;
  startBtn.disabled = true;

  setStatus("Solicitando acceso a la cámara…");
  try {
    await initCamera();
  } catch (err) {
    console.error("Error de cámara:", err);
    startBtn.disabled = false;

    if (isAutoAttempt) {
      setStatus("Haz clic en el botón 'Cámara' para comenzar.");
      return;
    }

    let msg = "No se pudo acceder a la cámara: ";
    if (window.location.protocol === "file:") {
      msg = "La cámara no funciona si abres este archivo directamente. Ejecuta iniciar_servidor.bat y abre http://localhost:8000.";
    } else if (!window.isSecureContext && !(window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "[::1]")) {
      msg = "Error: El navegador requiere un entorno seguro. Usa http://localhost:8000 desde el servidor local del proyecto.";
    } else if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
      msg += "Permiso denegado. Haz clic en el candado/icono a la izquierda de la barra de URL y permite el acceso a la cámara.";
    } else if (err && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")) {
      msg += "No se detectó ninguna cámara conectada en tu equipo.";
    } else if (err && (err.name === "NotReadableError" || err.name === "TrackStartError")) {
      msg += "La cámara está ocupada por otra aplicación (Zoom, Teams, etc.). Ciérrala y reintenta.";
    } else {
      msg += (err && err.message ? err.message : "revisa los permisos de cámara y vuelve a intentarlo.");
    }
    setStatus(msg);
    return;
  }

  // Arrancar loop de renderizado de inmediato para Sistema B
  running = true;
  cameraBtnWrapper.classList.add("is-live");
  requestAnimationFrame(renderLoop);

  setStatus("Cámara activa. Cargando modelo de manos (MediaPipe Hands)…");
  try {
    await initHands();
    setStatus("Listo. Ambos sistemas están activos: apunta con el índice para dibujar.");
  } catch (err) {
    console.error(err);
    setStatus(
  "El Color está activo. El Lienzo no pudo cargar el modelo de manos " +
        "(revisa tu conexión a internet y recarga la página).",
    );
    idleHintA.textContent = "modelo de manos no disponible";
  }
}

function setStatus(text) {
  statusMsg.textContent = text;
}

async function initCamera() {
  if (window.location.protocol === "file:") {
    throw new Error("La cámara no funciona si abres el archivo directamente. Usa http://localhost:8000.");
  }

  if (!window.isSecureContext && !(window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "[::1]")) {
    throw new Error("El navegador requiere un contexto seguro. Abre la página desde http://localhost:8000.");
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("API de medios no disponible en este contexto. Usa http://localhost:8000 en vez de file://.");
  }

  // Pedir primero una cámara genérica: antes del permiso, los navegadores
  // suelen ocultar los deviceId y sus etiquetas.
  const videoConstraints = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    facingMode: "user",
  };

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
  } catch (err) {
    console.warn("Fallo con restricciones específicas, reintentando con básicas:", err);
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  }

  // 2. Si la cámara conectada resultó ser la infrarroja (IR), cambiar a la cámara RGB
  try {
    const track = stream.getVideoTracks()[0];
    const trackLabel = (track && track.label ? track.label : "").toLowerCase();
    if (trackLabel.includes("ir") || trackLabel.includes("infra") || trackLabel.includes("hello")) {
      console.warn("Cámara IR detectada (" + trackLabel + "). Buscando cámara RGB alternativa...");
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const rgbDevice = allDevices
        .filter((d) => d.kind === "videoinput")
        .find((d) => {
          const l = (d.label || "").toLowerCase();
          return !l.includes("ir") && !l.includes("infra") && !l.includes("hello");
        });
      if (rgbDevice && rgbDevice.deviceId) {
        track.stop();
        stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: rgbDevice.deviceId }, width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      }
    }
  } catch (e) {
    console.warn("Error al cambiar de cámara IR a RGB:", e);
  }

  video.srcObject = stream;

  // Esperar a que el video cargue metadatos y dimensiones válidas sin riesgo de colgarse
  await new Promise((resolve) => {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      return resolve();
    }
    const onReady = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        video.removeEventListener("loadedmetadata", onReady);
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("canplay", onReady);
        resolve();
      }
    };
    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("loadeddata", onReady);
    video.addEventListener("canplay", onReady);
    setTimeout(resolve, 1500);
  });

  try {
    await video.play();
  } catch (err) {
    console.warn("video.play diferido:", err);
  }

  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvasA.width = w;
  canvasA.height = h;
  canvasB.width = w;
  canvasB.height = h;
}

async function initHands() {
  const { HandLandmarker, FilesetResolver } = await import(VISION_BUNDLE_URL);
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
  } catch (err) {
    console.warn("Fallo delegate GPU para MediaPipe Hands, reintentando con CPU…", err);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
  }
}

// ---------------------------------------------------------------------------
// Loop principal
// ---------------------------------------------------------------------------

function renderLoop(timestampMs) {
  if (!running) return;

  try {
    if (video.readyState >= 2 && video.videoWidth > 0) {
      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;

        let result = null;
        if (handLandmarker) {
          try {
            result = handLandmarker.detectForVideo(video, timestampMs);
          } catch (mpErr) {
            console.warn("Advertencia en MediaPipe detectForVideo:", mpErr);
          }
        }
        drawSystemA(result);
        drawSystemB();
      }
    }
  } catch (err) {
    console.error("Error en renderLoop:", err);
  }

  requestAnimationFrame(renderLoop);
}

// ---------------------------------------------------------------------------
// SISTEMA A — Visión Gestual y Trazado (MediaPipe Hands)
// ---------------------------------------------------------------------------

function drawSystemA(result) {
  const w = canvasA.width;
  const h = canvasA.height;
  const t = performance.now() / 1000;

  // Fondo gris claro base
  ctxA.fillStyle = "#f5f5f5";
  ctxA.fillRect(0, 0, w, h);

  // 1. Dibujar todos los trazos previos almacenados en el lienzo
  drawAllPaths(ctxA);

  const landmarks = result && result.landmarks && result.landmarks[0];

  if (!landmarks) {
    const withinTrackingGrace = currentPath && smoothIndexPos && performance.now() - lastIndexSeenAt < TIP_TRACKING_GRACE_MS;
    if (withinTrackingGrace) {
      idleHintA.style.opacity = "0";
      gestureBadge.textContent = "✏️ DIBUJANDO (siguiendo la punta)";
      gestureBadge.className = "gesture-badge is-drawing";
      statA.textContent = "manteniendo el trazo";
        drawIndexCursor(ctxA, smoothIndexPos.x, smoothIndexPos.y, true, targetHex, t);
      return;
    }

    idleHintA.style.opacity = "1";
    gestureBadge.textContent = "Esperando mano…";
    gestureBadge.className = "gesture-badge";
    statA.textContent = "sin mano detectada";
    smoothIndexPos = null;
    currentPath = null;
    drawIdlePulse(ctxA, w, h, t);
    return;
  }

  idleHintA.style.opacity = "0";

  // Identificar punta del dedo índice (Landmark 8)
  const indexTip = landmarks[8];
  if (!indexTip) {
    gestureBadge.textContent = "Esperando punta del índice…";
    gestureBadge.className = "gesture-badge";
    statA.textContent = "punta del índice no visible";
    if (performance.now() - lastIndexSeenAt >= TIP_TRACKING_GRACE_MS) {
      smoothIndexPos = null;
      currentPath = null;
    }
    return;
  }

  const targetX = clamp(indexTip.x * w, 0, w);
  const targetY = clamp(indexTip.y * h, 0, h);
  lastIndexSeenAt = performance.now();

  // Suavizado responsivo: sigue la punta sin perder continuidad.
  if (!smoothIndexPos) {
    smoothIndexPos = { x: targetX, y: targetY };
  } else {
    smoothIndexPos.x = smoothIndexPos.x * 0.75 + targetX * 0.25;
    smoothIndexPos.y = smoothIndexPos.y * 0.75 + targetY * 0.25;
  }

  // La punta se sigue siempre, pero solo el gesto de señalar permite dibujar.
  const pointingNow = isPointingGesture(landmarks);
  if (pointingNow) lastPointingAt = performance.now();
  isDrawing = pointingNow || (currentPath && performance.now() - lastPointingAt < GESTURE_TRACKING_GRACE_MS);

  if (isDrawing) {
    gestureBadge.textContent = "✏️ DIBUJANDO (Punta del índice)";
    gestureBadge.className = "gesture-badge is-drawing";
    statA.textContent = "punta del índice siguiendo la cámara";

    // Gestionar puntos del trazo actual
    if (!currentPath) {
      currentPath = {
        color: targetHex,
        width: 5,
        points: [{ x: smoothIndexPos.x, y: smoothIndexPos.y }],
      };
      paths.push(currentPath);
    } else {
      const lastPoint = currentPath.points[currentPath.points.length - 1];
      const dist = Math.hypot(smoothIndexPos.x - lastPoint.x, smoothIndexPos.y - lastPoint.y);
      if (dist >= 0.5) {
        // Interpolar posiciones evita cortes cuando el dedo se mueve rápido.
        const steps = Math.max(1, Math.ceil(dist / 1.5));
        for (let step = 1; step <= steps; step++) {
          const progress = step / steps;
          currentPath.points.push({
            x: lastPoint.x + (smoothIndexPos.x - lastPoint.x) * progress,
            y: lastPoint.y + (smoothIndexPos.y - lastPoint.y) * progress,
          });
        }
      }
    }
  } else {
    gestureBadge.textContent = "✋ EN ESPERA (Índice cerrado)";
    gestureBadge.className = "gesture-badge is-moving";
    statA.textContent = "punta localizada, sin trazar";
    currentPath = null;
  }

  // Mostrar la estructura completa de la mano y destacar la punta del índice
  drawHandSkeleton(ctxA, landmarks, w, h);
  drawIndexCursor(ctxA, smoothIndexPos.x, smoothIndexPos.y, isDrawing, targetHex, t);
}

// Determina si la mano está realizando el gesto de señalar/apuntar con el índice
function isPointingGesture(landmarks) {
  const angleAt = (a, b, c) => {
    const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
    const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
    const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
    const magnitude = Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z);
    return magnitude ? (Math.acos(clamp(dot / magnitude, -1, 1)) * 180) / Math.PI : 0;
  };

  const isFingerExtended = (mcp, pip, dip, tip) =>
    angleAt(landmarks[mcp], landmarks[pip], landmarks[dip]) > 145 &&
    angleAt(landmarks[pip], landmarks[dip], landmarks[tip]) > 145;

  const isIndexExtended = isFingerExtended(5, 6, 7, 8);
  const isMiddleCurled = !isFingerExtended(9, 10, 11, 12);
  const isRingCurled = !isFingerExtended(13, 14, 15, 16);
  const isPinkyCurled = !isFingerExtended(17, 18, 19, 20);

  // Para dibujar: índice extendido y al menos 2 de los 3 dedos restantes recogidos
  const curledCount = (isMiddleCurled ? 1 : 0) + (isRingCurled ? 1 : 0) + (isPinkyCurled ? 1 : 0);
  return isIndexExtended && curledCount >= 2;
}

// Dibuja todos los trazos acumulados en el lienzo
function drawAllPaths(ctx) {
  if (!paths.length) return;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  paths.forEach((path) => {
    if (!path.points || path.points.length < 2) return;

    ctx.save();
    ctx.strokeStyle = path.color;
    ctx.lineWidth = path.width || 5;
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;

    ctx.beginPath();
    ctx.moveTo(path.points[0].x, path.points[0].y);

    for (let i = 1; i < path.points.length - 1; i++) {
      const xc = (path.points[i].x + path.points[i + 1].x) / 2;
      const yc = (path.points[i].y + path.points[i + 1].y) / 2;
      ctx.quadraticCurveTo(path.points[i].x, path.points[i].y, xc, yc);
    }

    const lastIdx = path.points.length - 1;
    ctx.lineTo(path.points[lastIdx].x, path.points[lastIdx].y);
    ctx.stroke();
    ctx.restore();
  });
}

// Representación visual anatómica de la mano
function drawHandSkeleton(ctx, landmarks, w, h) {
  ctx.save();

  // Conexiones óseas
  HAND_CONNECTIONS.forEach(([ia, ib]) => {
    const a = landmarks[ia];
    const b = landmarks[ib];
    if (!a || !b) return;

    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Nodos articulares
  landmarks.forEach((p, idx) => {
    const color = FINGER_COLORS[idx] || "#ffffff";
    const r = idx === 8 ? 5 : 3;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = idx === 8 ? 0.9 : 0.45;
    ctx.fill();
  });

  ctx.restore();
}

function drawIndexFinger(ctx, landmarks, w, h, color, drawing) {
  const indexPoints = [5, 6, 7, 8].map((index) => landmarks[index]).filter(Boolean);
  if (indexPoints.length < 2) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(indexPoints[0].x * w, indexPoints[0].y * h);
  indexPoints.slice(1).forEach((point) => {
    ctx.lineTo(point.x * w, point.y * h);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = drawing ? 5 : 3;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.globalAlpha = drawing ? 0.95 : 0.7;
  ctx.shadowColor = color;
  ctx.shadowBlur = drawing ? 14 : 7;
  ctx.stroke();

  indexPoints.forEach((point, position) => {
    ctx.beginPath();
    ctx.arc(point.x * w, point.y * h, position === indexPoints.length - 1 ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = position === indexPoints.length - 1 ? 1 : 0.85;
    ctx.fill();
  });
  ctx.restore();
}

// Cursor interactivo en la punta del dedo índice
function drawIndexCursor(ctx, x, y, drawing, color, t) {
  ctx.save();
  const pulse = Math.sin(t * 8) * 2;
  const radius = drawing ? 9 + pulse : 6;

  // Anillo exterior brillante
  ctx.beginPath();
  ctx.arc(x, y, radius + 6, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.globalAlpha = drawing ? 0.85 : 0.4;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = drawing ? 18 : 6;
  ctx.stroke();

  // Centro sólido
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.globalAlpha = 1;
  ctx.fill();

  ctx.restore();
}

function drawIdlePulse(ctx, w, h, t) {
  const cx = w / 2, cy = h / 2;
  ctx.save();
  for (let i = 0; i < 3; i++) {
    const phase = t * 0.9 + i * 0.7;
    const r = 20 + ((phase * 40) % 130);
    const alpha = clamp(1 - r / 150, 0, 0.4);
    ctx.beginPath();
    ctx.strokeStyle = `rgba(255, 209, 102, ${alpha})`;
    ctx.lineWidth = 1.5;
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// SISTEMA B — Reconocimiento y Segmentación de Color (Visión Artificial)
// ---------------------------------------------------------------------------

function drawSystemB() {
  const w = canvasB.width;
  const h = canvasB.height;

  // Muestrear imagen de video en resolución reducida para alto rendimiento
  ctxHidden.drawImage(video, 0, 0, SAMPLE_COLS, SAMPLE_ROWS);
  const frame = ctxHidden.getImageData(0, 0, SAMPLE_COLS, SAMPLE_ROWS).data;

  // Fondo oscuro con leve persistencia
  ctxB.fillStyle = "rgba(5, 4, 10, 0.35)";
  ctxB.fillRect(0, 0, w, h);

  const cellW = w / SAMPLE_COLS;
  const cellH = h / SAMPLE_ROWS;
  const totalPixels = SAMPLE_COLS * SAMPLE_ROWS;
  let matchingCount = 0;

  // Segmentación cromática en espacio HSV
  for (let i = 0; i < totalPixels; i++) {
    const px = i * 4;
    const r = frame[px];
    const g = frame[px + 1];
    const b = frame[px + 2];

    const hsv = rgbToHsv(r, g, b);
    const sim = calculateColorSimilarity(hsv, targetHsv);

    // Si la similitud supera el umbral de detección
    if (sim > 0.65) {
      matchingCount++;
      const col = i % SAMPLE_COLS;
      const row = Math.floor(i / SAMPLE_COLS);

      const alpha = (sim - 0.65) / 0.35;
      const posX = col * cellW;
      const posY = row * cellH;

      ctxB.fillStyle = targetHex;
      ctxB.globalAlpha = alpha * 0.9;
      ctxB.fillRect(posX, posY, Math.max(1, cellW), Math.max(1, cellH));
    }
  }

  ctxB.globalAlpha = 1;

  // Reportar porcentaje de coincidencia en la escena
  const matchPercent = ((matchingCount / totalPixels) * 100).toFixed(1);
  statB.textContent = `${matchPercent}% de la escena coincide con ${targetHex}`;
  detectionBar.style.height = `${matchPercent}%`;
  detectionPercent.textContent = `${matchPercent}%`;
  detectionBarTrack.setAttribute("aria-valuenow", matchPercent);

  // Dibujar mira central de muestreo
  drawSamplingReticle(ctxB, w, h, targetHex);

  // Animación visual de retroalimentación de clic
  if (rippleFeedback) {
    drawRipple(ctxB);
  }
}

// Cálculo de similitud cromática perceptual (0 a 1)
function calculateColorSimilarity(hsv1, hsv2) {
  // Diferencia angular en matiz (Hue, circular de 0 a 1)
  let dh = Math.abs(hsv1.h - hsv2.h);
  if (dh > 0.5) dh = 1 - dh;
  const hueSim = Math.max(0, 1 - dh * 4.5);

  // Diferencia en saturación y brillo
  const ds = Math.abs(hsv1.s - hsv2.s);
  const satSim = Math.max(0, 1 - ds * 2.2);

  const dv = Math.abs(hsv1.v - hsv2.v);
  const valSim = Math.max(0, 1 - dv * 2.0);

  // Ponderación: el matiz predomina, salvo en colores muy desaturados
  if (hsv2.s < 0.2) {
    return satSim * 0.4 + valSim * 0.6;
  }
  return hueSim * 0.65 + satSim * 0.2 + valSim * 0.15;
}

// Mira/Retícula central para colocar el objeto a muestrear
function drawSamplingReticle(ctx, w, h, color) {
  const cx = w / 2;
  const cy = h / 2;
  const size = 56;
  const arm = 14;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  // Esquinas de la mira
  // Superior Izquierda
  ctx.beginPath();
  ctx.moveTo(cx - size, cy - size + arm);
  ctx.lineTo(cx - size, cy - size);
  ctx.lineTo(cx - size + arm, cy - size);
  ctx.stroke();

  // Superior Derecha
  ctx.beginPath();
  ctx.moveTo(cx + size - arm, cy - size);
  ctx.lineTo(cx + size, cy - size);
  ctx.lineTo(cx + size, cy - size + arm);
  ctx.stroke();

  // Inferior Izquierda
  ctx.beginPath();
  ctx.moveTo(cx - size, cy + size - arm);
  ctx.lineTo(cx - size, cy + size);
  ctx.lineTo(cx - size + arm, cy + size);
  ctx.stroke();

  // Inferior Derecha
  ctx.beginPath();
  ctx.moveTo(cx + size - arm, cy + size);
  ctx.lineTo(cx + size, cy + size);
  ctx.lineTo(cx + size, cy + size - arm);
  ctx.stroke();

  // Cruz central pequeña
  ctx.beginPath();
  ctx.moveTo(cx - 5, cy);
  ctx.lineTo(cx + 5, cy);
  ctx.moveTo(cx, cy - 5);
  ctx.lineTo(cx, cy + 5);
  ctx.stroke();

  ctx.restore();
}

// Muestrea el color promedio del objeto ubicado en la mira central
function sampleCenterColor() {
  if (!running) {
    setStatus("Primero activa la cámara para capturar un color.");
    return;
  }

  const sampleBoxSize = 14;
  const startX = Math.floor((SAMPLE_COLS - sampleBoxSize) / 2);
  const startY = Math.floor((SAMPLE_ROWS - sampleBoxSize) / 2);

  const imgData = ctxHidden.getImageData(startX, startY, sampleBoxSize, sampleBoxSize).data;
  let totalR = 0, totalG = 0, totalB = 0;
  const total = sampleBoxSize * sampleBoxSize;

  for (let i = 0; i < total; i++) {
    const px = i * 4;
    totalR += imgData[px];
    totalG += imgData[px + 1];
    totalB += imgData[px + 2];
  }

  const avgR = Math.round(totalR / total);
  const avgG = Math.round(totalG / total);
  const avgB = Math.round(totalB / total);

  setTargetColorRgb(avgR, avgG, avgB);
  triggerRipple(canvasB.width / 2, canvasB.height / 2);
  setStatus(`Color capturado: ${targetHex}. Los nuevos trazos se pintarán con este color.`);
}

// Muestrea el color del píxel donde se hizo clic en el Canvas B
function sampleColorFromCanvasClick(event) {
  const rect = canvasB.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;

  // Tomar en cuenta la inversión de espejo del CSS (transform: scaleX(-1))
  const mirroredX = rect.width - clickX;
  const sampleX = Math.floor((mirroredX / rect.width) * SAMPLE_COLS);
  const sampleY = Math.floor((clickY / rect.height) * SAMPLE_ROWS);

  const pixel = ctxHidden.getImageData(
    clamp(sampleX, 0, SAMPLE_COLS - 1),
    clamp(sampleY, 0, SAMPLE_ROWS - 1),
    1,
    1,
  ).data;

  setTargetColorRgb(pixel[0], pixel[1], pixel[2]);
  triggerRipple(clickX, clickY);
  setStatus(`Color seleccionado del punto: ${targetHex}.`);
}

function triggerRipple(x, y) {
  rippleFeedback = { x, y, radius: 5, maxRadius: 40, alpha: 1 };
}

function drawRipple(ctx) {
  if (!rippleFeedback) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(rippleFeedback.x, rippleFeedback.y, rippleFeedback.radius, 0, Math.PI * 2);
  ctx.strokeStyle = targetHex;
  ctx.globalAlpha = rippleFeedback.alpha;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.restore();

  rippleFeedback.radius += 2.5;
  rippleFeedback.alpha *= 0.88;
  if (rippleFeedback.alpha < 0.05) {
    rippleFeedback = null;
  }
}

// ---------------------------------------------------------------------------
// Actualización global del color activo
// ---------------------------------------------------------------------------

function setTargetColorRgb(r, g, b) {
  targetRgb = { r, g, b };
  targetHsv = rgbToHsv(r, g, b);
  targetHex = rgbToHex(r, g, b);
  updateColorUI();
}

function setTargetColorFromHex(hex) {
  const rgb = hexToRgb(hex);
  if (rgb) {
    setTargetColorRgb(rgb.r, rgb.g, rgb.b);
  }
}

function updateColorUI() {
  colorPreview.style.backgroundColor = targetHex;
  colorHex.textContent = targetHex;
  detectionChart.style.setProperty("--bar-color", targetHex);

  // Actualizar chip activo en la barra de presets si coincide
  presetChips.forEach((chip) => {
    if (chip.getAttribute("data-color").toLowerCase() === targetHex.toLowerCase()) {
      chip.classList.add("active");
    } else {
      chip.classList.remove("active");
    }
  });
}

// ---------------------------------------------------------------------------
// Funciones auxiliares de conversión cromática y matemática
// ---------------------------------------------------------------------------

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  const s = max === 0 ? 0 : d / max;
  const v = max;

  if (d !== 0) {
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
    }
  }
  return { h, s, v };
}

function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("").toUpperCase();
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  if (clean.length === 6) {
    return {
      r: parseInt(clean.substring(0, 2), 16),
      g: parseInt(clean.substring(2, 4), 16),
      b: parseInt(clean.substring(4, 6), 16),
    };
  }
  return null;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}
