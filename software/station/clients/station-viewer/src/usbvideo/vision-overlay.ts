import type { VisionDetection } from './vision-types';

export interface ImageLayoutRect {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
}

export function getImageLayoutRect(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
  fit: 'contain' | 'cover',
): ImageLayoutRect {
  if (containerWidth <= 0 || containerHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return { x: 0, y: 0, width: 0, height: 0, scale: 1 };
  }

  const containerAspect = containerWidth / containerHeight;
  const imageAspect = imageWidth / imageHeight;

  if (fit === 'cover') {
    const scale =
      containerAspect > imageAspect
        ? containerWidth / imageWidth
        : containerHeight / imageHeight;
    const width = imageWidth * scale;
    const height = imageHeight * scale;
    return {
      x: (containerWidth - width) / 2,
      y: (containerHeight - height) / 2,
      width,
      height,
      scale,
    };
  }

  const scale =
    containerAspect > imageAspect
      ? containerHeight / imageHeight
      : containerWidth / imageWidth;
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: (containerWidth - width) / 2,
    y: (containerHeight - height) / 2,
    width,
    height,
    scale,
  };
}

function mapPoint(
  x: number,
  y: number,
  layout: ImageLayoutRect,
): [number, number] {
  return [layout.x + x * layout.scale, layout.y + y * layout.scale];
}

function drawObb(
  ctx: CanvasRenderingContext2D,
  detection: VisionDetection,
  layout: ImageLayoutRect,
  color: string,
) {
  const [centerX, centerY] = mapPoint(detection.center_xy[0], detection.center_xy[1], layout);
  const width = detection.size_wh[0] * layout.scale;
  const height = detection.size_wh[1] * layout.scale;
  const angleRad = (detection.angle_deg * Math.PI) / 180;

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(angleRad);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(-width / 2, -height / 2, width, height);

  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(width / 2, 0);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

export function drawDetectionOverlay(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  detections: VisionDetection[],
  fit: 'contain' | 'cover',
  sourceWidth: number,
  sourceHeight: number,
  inferenceFps?: number,
  error?: string | null,
) {
  const containerWidth = canvas.clientWidth;
  const containerHeight = canvas.clientHeight;
  if (containerWidth <= 0 || containerHeight <= 0) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(containerWidth * dpr);
  canvas.height = Math.round(containerHeight * dpr);

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, containerWidth, containerHeight);

  const imageWidth = sourceWidth > 0 ? sourceWidth : image.naturalWidth;
  const imageHeight = sourceHeight > 0 ? sourceHeight : image.naturalHeight;
  const layout = getImageLayoutRect(containerWidth, containerHeight, imageWidth, imageHeight, fit);

  const classColors: Record<string, string> = {
    'black block': '#22c55e',
    'red bull': '#ef4444',
  };
  const fallbackColors = ['#38bdf8', '#f97316', '#eab308', '#a855f7'];
  detections.forEach((detection, index) => {
    const color = classColors[detection.class_name] ?? fallbackColors[index % fallbackColors.length];
    drawObb(ctx, detection, layout, color);

    const [labelX, labelY] = mapPoint(
      detection.bbox_xyxy[0],
      Math.max(0, detection.bbox_xyxy[1] - 4),
      layout,
    );
    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`;
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
    ctx.fillRect(labelX, labelY - 16, textWidth + 8, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, labelX + 4, labelY - 3);
  });

  ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
  ctx.fillRect(8, containerHeight - 28, 280, 20);
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  const status = error
    ? `Vision error: ${error}`
    : `Local vision: ${detections.length} det${inferenceFps != null ? ` | ${inferenceFps.toFixed(1)} fps` : ''}`;
  ctx.fillText(status, 12, containerHeight - 14);
}
