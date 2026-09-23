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

// Estado de color (Sistema B)
let targetRgb = { r: 255, g: 77, b: 109 };
let targetHsv = rgbToHsv(255, 77, 109);
let targetHex = "#FF4D6D";
let rippleFeedback = null;

// Configurar dimensiones de muestreo
hiddenSample.width = SAMPLE_COLS;
hiddenSample.height = SAMPLE_ROWS;
// ---------------------------------------------------------------------------
// Inicialización y eventos
// ---------------------------------------------------------------------------

startBtn.addEventListener("click", start);
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

// ---------------------------------------------------------------------------
// Arranque de cámara y modelo
// ---------------------------------------------------------------------------

async function start() {
  startBtn.disabled = true;

  setStatus("Solicitando acceso a la cámara…");
  try {
    await initCamera();
  } catch (err) {
    console.error(err);
    setStatus(
      "No se pudo acceder a la cámara: " +
        (err && err.message ? err.message : "revisa los permisos de cámara y vuelve a intentarlo."),
    );
    startBtn.disabled = false;
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
      "El Sistema B (Color) está activo. El Sistema A no pudo cargar el modelo de manos " +
        "(revisa tu conexión a internet y recarga la página).",
    );
    idleHintA.textContent = "modelo de manos no disponible";
  }
}

function setStatus(text) {
  statusMsg.textContent = text;
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  await new Promise((resolve) => {
    if (video.readyState >= 2) return resolve();
    video.onloadedmetadata = () => resolve();
  });

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

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const result = handLandmarker ? handLandmarker.detectForVideo(video, timestampMs) : null;
    drawSystemA(result);
    drawSystemB();
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

  // Fondo oscuro base
  ctxA.fillStyle = "#07060c";
  ctxA.fillRect(0, 0, w, h);

  // 1. Dibujar todos los trazos previos almacenados en el lienzo
  drawAllPaths(ctxA);

  const landmarks = result && result.landmarks && result.landmarks[0];

  if (!landmarks) {
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
  const targetX = indexTip.x * w;
  const targetY = indexTip.y * h;

  // Suavizado exponencial para amortiguar vibraciones (jitter)
  if (!smoothIndexPos) {
    smoothIndexPos = { x: targetX, y: targetY };
  } else {
    smoothIndexPos.x = smoothIndexPos.x * 0.35 + targetX * 0.65;
    smoothIndexPos.y = smoothIndexPos.y * 0.35 + targetY * 0.65;
  }

  // 2. Reconocimiento del gesto: ¿está apuntando solo con el índice?
  isDrawing = isPointingGesture(landmarks);

  if (isDrawing) {
    gestureBadge.textContent = "✏️ DIBUJANDO (Índice activo)";
    gestureBadge.className = "gesture-badge is-drawing";
    statA.textContent = "gesto de dibujo detectado";

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
      if (dist >= 2) {
        currentPath.points.push({ x: smoothIndexPos.x, y: smoothIndexPos.y });
      }
    }
  } else {
    gestureBadge.textContent = "✋ MOVIENDO (Mano abierta)";
    gestureBadge.className = "gesture-badge is-moving";
    statA.textContent = "reposicionando sin trazar";
    currentPath = null;
  }

  // 3. Dibujar esqueleto anatómico de la mano de forma sutil
  drawHandSkeleton(ctxA, landmarks, w, h);

  // 4. Dibujar cursor brillante interactivo en la punta del dedo índice
  drawIndexCursor(ctxA, smoothIndexPos.x, smoothIndexPos.y, isDrawing, targetHex, t);
}

// Determina si la mano está realizando el gesto de señalar/apuntar con el índice
function isPointingGesture(landmarks) {
  const wrist = landmarks[0];

  const distToWrist = (idx) => {
    const p = landmarks[idx];
    return Math.hypot(p.x - wrist.x, p.y - wrist.y, (p.z ?? 0) - (wrist.z ?? 0));
  };

  // Dedo índice (MCP: 5, PIP: 6, TIP: 8)
  const indexTipDist = distToWrist(8);
  const indexPipDist = distToWrist(6);
  const indexMcpDist = distToWrist(5);
  const isIndexExtended = indexTipDist > indexPipDist * 1.15 && indexTipDist > indexMcpDist * 1.3;

  // Dedo medio (MCP: 9, PIP: 10, TIP: 12)
  const middleTipDist = distToWrist(12);
  const middlePipDist = distToWrist(10);
  const isMiddleCurled = middleTipDist < middlePipDist * 1.18;

  // Dedo anular (MCP: 13, PIP: 14, TIP: 16)
  const ringTipDist = distToWrist(16);
  const ringPipDist = distToWrist(14);
  const isRingCurled = ringTipDist < ringPipDist * 1.18;

  // Dedo meñique (MCP: 17, PIP: 18, TIP: 20)
  const pinkyTipDist = distToWrist(20);
  const pinkyPipDist = distToWrist(18);
  const isPinkyCurled = pinkyTipDist < pinkyPipDist * 1.18;

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
    ctx.shadowColor = path.color;
    ctx.shadowBlur = 10;

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

  // Etiqueta explicativa
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(241, 236, 247, 0.7)";
  ctx.textAlign = "center";
  ctx.fillText("ZONA DE MUESTREO", cx, cy + size + 18);

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
