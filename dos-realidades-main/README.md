# Dos realidades

Ejercicio 02 del curso (DPPI 2026), sobre visión artificial y representación. La idea era tomar una sola cámara y usarla para armar dos maneras completamente distintas de "ver" lo mismo.

**Demo:** https://fefeliperoar.github.io/dos-realidades/  
**Repo:** https://github.com/fefeliperoar/dos-realidades  

## De qué se trata

Hay dos sistemas corriendo al mismo tiempo, con la misma cámara. Ninguno de los dos muestra la imagen de la cámara tal cual — cada uno se queda solo con el dato que le importa y lo dibuja a su manera. Por eso terminan pareciendo dos realidades distintas aunque estén mirando la misma escena.

**Sistema A — Gesto y Trazado (MediaPipe Hands).**  
Usa MediaPipe Hand Landmarker para identificar la anatomía de la mano y sus 21 puntos articulares. Reconoce el gesto humano de apuntar con el dedo índice (índice extendido mientras los otros dedos permanecen doblados) para convertir el movimiento de la mano en un lienzo de dibujo continuo en el aire. Si abres toda la mano, el sistema entra en modo reposicionamiento/cursor sin dibujar.

**Sistema B — Reconocimiento Cromático (Visión por Computadora).**  
Este sistema no tiene noción de manos ni de anatomía: solo observa la luz y el espectro cromático de los píxeles. Permite seleccionar un color específico (centrando un objeto en la mira y pulsando "Capturar color", haciendo clic sobre la imagen o eligiendo un preset) y resalta en tiempo real dónde aparece ese color en la escena física.

**El diálogo entre ambos:**  
El color que el Sistema B aísla de la realidad física se transfiere directamente al Sistema A como la tinta con la que el dedo índice dibuja en el espacio.

## Cómo probarlo

El `index.html` no se puede abrir directo con doble clic porque el script usa módulos de JS y requiere contexto seguro para la cámara. Hay que levantar un servidor local desde la carpeta, por ejemplo:

```bash
python -m http.server 8000
```

y entrar a `http://localhost:8000`. Pasos para interactuar:
1. Haz clic en el botón **"Cámara"** y concede el permiso en el navegador.
2. Coloca tu mano frente a la cámara: extiende solo el dedo índice para **dibujar** en el aire. Abre la mano para moverte sin trazar.
3. Con un objeto de color (una tapa, un cuaderno, ropa, etc.), colócalo en el centro del Sistema B y presiona **"Capturar color del centro"** (o haz clic sobre el objeto en el canvas derecho).
4. Verás cómo el Sistema B resalta ese color y el trazo del dedo índice en el Sistema A cambia inmediatamente a ese nuevo color.
5. Puedes pulsar **"Limpiar dibujo"** para reiniciar el lienzo cuando quieras.

## Reflexión

Frente a la cámara ocurre una sola escena, pero cada sistema encuentra algo distinto en ella. Uno reconoce una mano y un gesto intencional; el otro simplemente observa dónde coincide una frecuencia cromática. Ninguno está equivocado, pero ninguno puede verlo todo.

Merleau-Ponty planteaba que nuestra percepción está ligada a las posibilidades y límites de nuestro cuerpo. Con las máquinas ocurre algo parecido: aquello que pueden percibir depende de cómo fueron construidas y de qué les enseñamos a buscar.

Kosuth, por otro lado, nos permite recordar que una representación nunca es aquello que representa. Los trazos y las manchas de color en la pantalla hablan de una persona y un objeto, pero no son la persona ni el objeto.

Tal vez lo interesante de construir una máquina que observa no sea preguntarnos cuánto puede ver, sino comenzar a reconocer todo aquello que, inevitablemente, deja fuera.

## Tecnologías

- MediaPipe Hand Landmarker (cargado dinámicamente vía CDN jsDelivr).
- Segmentación de color en espacio HSV y Canvas 2D con JavaScript nativo (vanilla JS).
- Sin frameworks ni dependencias de empaquetado (buildless).

---
Felipe · Ejercicio 02 — Dos realidades · DPPI 2026

