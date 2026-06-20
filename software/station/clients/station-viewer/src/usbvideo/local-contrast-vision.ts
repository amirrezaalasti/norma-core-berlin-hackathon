import type { CameraCalibrationData } from './camera-calibration';
import { withCameraWorkspaceUnits } from './camera-calibration';
import type { VisionDetection, WorkspaceCalibration } from './vision-types';
import { detectWorkspace, enrichDetectionsWithWorkspace, filterDetectionsInWorkspace } from './workspace-detection';

export const LOCAL_VISION_CLASSES = ['block', 'box', 'cube'] as const;

function luminance(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

function saturation(r: number, g: number, b: number): number {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  if (max <= 0) {
    return 0;
  }
  return (max - min) / max;
}

function isDarkPixel(r: number, g: number, b: number): boolean {
  return luminance(r, g, b) < 80 && saturation(r, g, b) < 0.3;
}

function otsuThreshold(gray: Uint8Array): number {
  const histogram = new Array<number>(256).fill(0);
  for (const value of gray) {
    histogram[value] += 1;
  }

  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i += 1) {
    sum += i * histogram[i];
  }

  let sumBackground = 0;
  let weightBackground = 0;
  let bestThreshold = 128;
  let bestVariance = 0;

  for (let threshold = 0; threshold < 256; threshold += 1) {
    weightBackground += histogram[threshold];
    if (weightBackground === 0) {
      continue;
    }

    const weightForeground = total - weightBackground;
    if (weightForeground === 0) {
      break;
    }

    sumBackground += threshold * histogram[threshold];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const variance =
      weightBackground *
      weightForeground *
      (meanBackground - meanForeground) *
      (meanBackground - meanForeground);

    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = threshold;
    }
  }

  return bestThreshold;
}

interface BlobBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  count: number;
}

interface RegionStats {
  darkRatio: number;
  meanLuminance: number;
  sampleCount: number;
}

function findBlobs(
  mask: Uint8Array,
  width: number,
  height: number,
): BlobBox[] {
  const visited = new Uint8Array(width * height);
  const blobs: BlobBox[] = [];

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index] || visited[index]) {
        continue;
      }

      const stack = [index];
      visited[index] = 1;
      const blob: BlobBox = {
        minX: x,
        minY: y,
        maxX: x,
        maxY: y,
        count: 0,
      };

      while (stack.length > 0) {
        const current = stack.pop();
        if (current == null) {
          continue;
        }

        const cy = Math.floor(current / width);
        const cx = current - cy * width;
        blob.count += 1;
        blob.minX = Math.min(blob.minX, cx);
        blob.minY = Math.min(blob.minY, cy);
        blob.maxX = Math.max(blob.maxX, cx);
        blob.maxY = Math.max(blob.maxY, cy);

        for (const [nx, ny] of [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ]) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          const neighborIndex = ny * width + nx;
          if (!mask[neighborIndex] || visited[neighborIndex]) {
            continue;
          }
          visited[neighborIndex] = 1;
          stack.push(neighborIndex);
        }
      }

      blobs.push(blob);
    }
  }

  return blobs;
}

function sampleRegionStats(
  imageData: ImageData,
  width: number,
  blob: BlobBox,
): RegionStats {
  let darkCount = 0;
  let luminanceSum = 0;
  let sampleCount = 0;
  const step = Math.max(1, Math.floor(Math.sqrt(blob.count) / 8));

  for (let y = blob.minY; y <= blob.maxY; y += step) {
    for (let x = blob.minX; x <= blob.maxX; x += step) {
      const offset = (y * width + x) * 4;
      const r = imageData.data[offset];
      const g = imageData.data[offset + 1];
      const b = imageData.data[offset + 2];
      sampleCount += 1;
      luminanceSum += luminance(r, g, b);
      if (isDarkPixel(r, g, b)) {
        darkCount += 1;
      }
    }
  }

  return {
    darkRatio: sampleCount > 0 ? darkCount / sampleCount : 0,
    meanLuminance: sampleCount > 0 ? luminanceSum / sampleCount : 255,
    sampleCount,
  };
}

function isBlackBlock(stats: RegionStats): boolean {
  if (stats.darkRatio >= 0.42 && stats.meanLuminance < 85) {
    return true;
  }
  return stats.meanLuminance < 65;
}

function isValidBlobSize(blob: BlobBox, width: number, height: number): boolean {
  const imageArea = width * height;
  const area = blob.count;
  const minRatio = Math.max(width, height) < 400 ? 0.0008 : 0.0015;
  return area >= imageArea * minRatio && area <= imageArea * 0.2;
}

function blobDistance(blob: BlobBox, centerX: number, centerY: number): number {
  const cx = (blob.minX + blob.maxX) / 2;
  const cy = (blob.minY + blob.maxY) / 2;
  return Math.hypot(cx - centerX, cy - centerY);
}

function blobToDetection(
  blob: BlobBox,
  className: string,
  confidence: number,
  scale: number,
): VisionDetection {
  const invScale = 1 / scale;
  const x1 = blob.minX * invScale;
  const y1 = blob.minY * invScale;
  const x2 = (blob.maxX + 1) * invScale;
  const y2 = (blob.maxY + 1) * invScale;
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const width = x2 - x1;
  const height = y2 - y1;

  return {
    class_name: className,
    confidence,
    bbox_xyxy: [x1, y1, x2, y2],
    center_xy: [centerX, centerY],
    size_wh: [width, height],
    angle_deg: width >= height ? 0 : 90,
  };
}

function dedupeDetections(detections: VisionDetection[]): VisionDetection[] {
  const kept: VisionDetection[] = [];
  for (const detection of detections.sort((a, b) => b.confidence - a.confidence)) {
    const overlaps = kept.some((existing) => {
      const [ax1, ay1, ax2, ay2] = existing.bbox_xyxy;
      const [bx1, by1, bx2, by2] = detection.bbox_xyxy;
      const x1 = Math.max(ax1, bx1);
      const y1 = Math.max(ay1, by1);
      const x2 = Math.min(ax2, bx2);
      const y2 = Math.min(ay2, by2);
      if (x2 <= x1 || y2 <= y1) {
        return false;
      }
      const intersection = (x2 - x1) * (y2 - y1);
      const areaA = Math.max(1, (ax2 - ax1) * (ay2 - ay1));
      const areaB = Math.max(1, (bx2 - bx1) * (by2 - by1));
      return intersection / (areaA + areaB - intersection) > 0.3;
    });
    if (!overlaps) {
      kept.push(detection);
    }
  }
  return kept;
}

function detectBlackBlocks(
  darkMask: Uint8Array,
  imageData: ImageData,
  width: number,
  height: number,
  scale: number,
  centerX: number,
  centerY: number,
): VisionDetection[] {
  return findBlobs(darkMask, width, height)
    .filter((blob) => isValidBlobSize(blob, width, height))
    .map((blob) => ({ blob, stats: sampleRegionStats(imageData, width, blob) }))
    .filter(({ stats }) => isBlackBlock(stats))
    .map(({ blob, stats }) => ({
      detection: blobToDetection(
        blob,
        'block',
        Math.min(0.97, 0.6 + stats.darkRatio * 0.35),
        scale,
      ),
      distance: blobDistance(blob, centerX, centerY),
    }))
    .sort((a, b) => a.distance - b.distance)
    .map((item) => item.detection);
}

export interface LocalVisionResult {
  detections: VisionDetection[];
  workspace: WorkspaceCalibration | null;
}

export function detectObjectsLocal(
  image: HTMLImageElement,
  maxAnalysisSize = 480,
  externalWorkspace: WorkspaceCalibration | null = null,
  manualWorkspace: WorkspaceCalibration | null = null,
  cameraCalibration: CameraCalibrationData | null = null,
): LocalVisionResult {
  if (image.naturalWidth <= 0 || image.naturalHeight <= 0) {
    return { detections: [], workspace: manualWorkspace ?? null };
  }

  const workspaceAnalysisSize = 960;
  const wsScale = Math.min(
    1,
    workspaceAnalysisSize / Math.max(image.naturalWidth, image.naturalHeight),
  );
  const blockScale = Math.min(
    1,
    maxAnalysisSize / Math.max(image.naturalWidth, image.naturalHeight),
  );

  const wsWidth = Math.max(1, Math.round(image.naturalWidth * wsScale));
  const wsHeight = Math.max(1, Math.round(image.naturalHeight * wsScale));
  const blockWidth = Math.max(1, Math.round(image.naturalWidth * blockScale));
  const blockHeight = Math.max(1, Math.round(image.naturalHeight * blockScale));

  const wsCanvas = document.createElement('canvas');
  wsCanvas.width = wsWidth;
  wsCanvas.height = wsHeight;
  const wsCtx = wsCanvas.getContext('2d');
  if (!wsCtx) {
    return { detections: [], workspace: null };
  }
  wsCtx.drawImage(image, 0, 0, wsWidth, wsHeight);
  const wsImageData = wsCtx.getImageData(0, 0, wsWidth, wsHeight);
  const detectedWorkspace = manualWorkspace ?? detectWorkspace(wsImageData, wsWidth, wsHeight, wsScale, externalWorkspace);
  const workspace = manualWorkspace ?? withCameraWorkspaceUnits(detectedWorkspace, cameraCalibration);

  const blockCanvas = document.createElement('canvas');
  blockCanvas.width = blockWidth;
  blockCanvas.height = blockHeight;
  const blockCtx = blockCanvas.getContext('2d');
  if (!blockCtx) {
    return { detections: [], workspace };
  }

  blockCtx.drawImage(image, 0, 0, blockWidth, blockHeight);
  const imageData = blockCtx.getImageData(0, 0, blockWidth, blockHeight);
  const gray = new Uint8Array(blockWidth * blockHeight);
  const darkMask = new Uint8Array(blockWidth * blockHeight);

  for (let i = 0; i < blockWidth * blockHeight; i += 1) {
    gray[i] = luminance(
      imageData.data[i * 4],
      imageData.data[i * 4 + 1],
      imageData.data[i * 4 + 2],
    );
  }

  const threshold = Math.min(otsuThreshold(gray) * 0.95, 140);
  for (let i = 0; i < gray.length; i += 1) {
    darkMask[i] = gray[i] < threshold ? 1 : 0;
  }

  const centerX = blockWidth / 2;
  const centerY = blockHeight / 2;

  const detections = filterDetectionsInWorkspace(
    enrichDetectionsWithWorkspace(
      dedupeDetections(
        detectBlackBlocks(darkMask, imageData, blockWidth, blockHeight, blockScale, centerX, centerY),
      ).slice(0, 4),
      workspace,
      manualWorkspace ? null : cameraCalibration,
    ),
    workspace,
  );

  return { detections, workspace };
}

/** @deprecated Use detectObjectsLocal — kept for callers that only need dark blobs. */
export function detectDarkObjectsLocal(
  image: HTMLImageElement,
  _className = 'block',
  maxAnalysisSize = 320,
): VisionDetection[] {
  return detectObjectsLocal(image, maxAnalysisSize).detections;
}
