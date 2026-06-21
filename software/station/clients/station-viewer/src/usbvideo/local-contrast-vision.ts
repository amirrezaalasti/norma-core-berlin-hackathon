import type { CameraCalibrationData } from './camera-calibration';
import { withCameraWorkspaceUnits } from './camera-calibration';
import { scaleManualWorkspaceToImage } from './manual-workspace-calibration';
import type { GripperTipPosition, VisionDetection, WorkspaceCalibration } from './vision-types';
import {
  detectWorkspace,
  enrichDetectionsWithWorkspace,
  filterDetectionsInWorkspace,
  gripperTipFromManualWorkspace,
  gripperTipFromPixel,
  pixelToBoardNormalized,
} from './workspace-detection';

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

function isYellowTapePixel(r: number, g: number, b: number): boolean {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const sat = max > 0 ? (max - min) / max : 0;
  return (
    sat > 0.22 &&
    r > 95 &&
    g > 75 &&
    b < 145 &&
    r > b + 20 &&
    g > b + 5 &&
    (r + g) / 2 > 105
  );
}

function isValidYellowBlob(blob: BlobBox, width: number, height: number): boolean {
  const imageArea = width * height;
  const area = blob.count;
  const boxW = blob.maxX - blob.minX + 1;
  const boxH = blob.maxY - blob.minY + 1;
  const aspect = Math.max(boxW, boxH) / Math.max(Math.min(boxW, boxH), 1);
  return (
    area >= imageArea * 0.0004 &&
    area <= imageArea * 0.08 &&
    aspect <= 4.5
  );
}

function detectYellowTape(
  imageData: ImageData,
  width: number,
  height: number,
  scale: number,
  hint: [number, number] | null,
): VisionDetection | null {
  const yellowMask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i += 1) {
    const offset = i * 4;
    const r = imageData.data[offset];
    const g = imageData.data[offset + 1];
    const b = imageData.data[offset + 2];
    yellowMask[i] = isYellowTapePixel(r, g, b) ? 1 : 0;
  }

  const hintPxX = hint ? hint[0] * scale : width * 0.35;
  const hintPxY = hint ? hint[1] * scale : height * 0.45;

  const candidates = findBlobs(yellowMask, width, height)
    .filter((blob) => isValidYellowBlob(blob, width, height))
    .map((blob) => {
      const centerX = (blob.minX + blob.maxX + 1) / 2;
      const centerY = (blob.minY + blob.maxY + 1) / 2;
      const dist = Math.hypot(centerX - hintPxX, centerY - hintPxY);
      const leftBias = centerX < width * 0.62 ? 0 : width * 0.15;
      return {
        blob,
        score: blob.count / Math.max(dist + leftBias, 1),
      };
    })
    .sort((a, b) => b.score - a.score);

  const best = candidates[0];
  if (!best) {
    return null;
  }

  return blobToDetection(best.blob, 'yellow_tape', 0.88, scale);
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
  gripperTip: GripperTipPosition | null;
}

export function detectObjectsLocal(
  image: HTMLImageElement,
  maxAnalysisSize = 480,
  externalWorkspace: WorkspaceCalibration | null = null,
  manualWorkspace: WorkspaceCalibration | null = null,
  cameraCalibration: CameraCalibrationData | null = null,
): LocalVisionResult {
  if (image.naturalWidth <= 0 || image.naturalHeight <= 0) {
    return { detections: [], workspace: manualWorkspace ?? null, gripperTip: null };
  }

  const scaledManual = manualWorkspace
    ? scaleManualWorkspaceToImage(
        manualWorkspace,
        image.naturalWidth,
        image.naturalHeight,
      )
    : null;
  const hasManual = Boolean(scaledManual?.corners_xy && scaledManual.calibration_source === 'manual');

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

  let workspace: WorkspaceCalibration | null = scaledManual;

  if (!hasManual) {
    const wsCanvas = document.createElement('canvas');
    wsCanvas.width = wsWidth;
    wsCanvas.height = wsHeight;
    const wsCtx = wsCanvas.getContext('2d');
    if (!wsCtx) {
      return { detections: [], workspace: null, gripperTip: null };
    }
    wsCtx.drawImage(image, 0, 0, wsWidth, wsHeight);
    const wsImageData = wsCtx.getImageData(0, 0, wsWidth, wsHeight);
    const detectedWorkspace = detectWorkspace(wsImageData, wsWidth, wsHeight, wsScale, externalWorkspace);
    workspace = withCameraWorkspaceUnits(detectedWorkspace, cameraCalibration);
  }

  const blockCanvas = document.createElement('canvas');
  blockCanvas.width = blockWidth;
  blockCanvas.height = blockHeight;
  const blockCtx = blockCanvas.getContext('2d');
  if (!blockCtx) {
    return { detections: [], workspace, gripperTip: null };
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

  const yellowTape = detectYellowTape(
    imageData,
    blockWidth,
    blockHeight,
    blockScale,
    scaledManual?.origin_xy ?? null,
  );

  let workingWorkspace = workspace;
  let gripperTip: GripperTipPosition | null = null;

  if (yellowTape && workingWorkspace) {
    const origin = yellowTape.center_xy;
    workingWorkspace = {
      ...workingWorkspace,
      origin_xy: origin,
      gripper_tip_set: true,
    };
    gripperTip = gripperTipFromPixel(
      origin[0],
      origin[1],
      workingWorkspace,
      'detected',
      yellowTape.confidence,
    );
  } else if (scaledManual?.gripper_tip_set && scaledManual.origin_xy && workingWorkspace) {
    gripperTip = gripperTipFromManualWorkspace(workingWorkspace);
  }

  const blockDetections = dedupeDetections(
    detectBlackBlocks(darkMask, imageData, blockWidth, blockHeight, blockScale, centerX, centerY),
  ).slice(0, 4);

  const canEnrichOffsets = Boolean(workingWorkspace && (hasManual || workingWorkspace.gripper_tip_set));

  const detections = canEnrichOffsets
    ? filterDetectionsInWorkspace(
        enrichDetectionsWithWorkspace(
          blockDetections,
          workingWorkspace!,
          hasManual ? null : cameraCalibration,
        ),
        workingWorkspace!,
      )
    : blockDetections.map((detection) => {
        if (!workingWorkspace) {
          return detection;
        }
        const board_xy = pixelToBoardNormalized(
          detection.center_xy[0],
          detection.center_xy[1],
          workingWorkspace,
        );
        return board_xy ? { ...detection, board_xy } : detection;
      });

  let enrichedYellow = yellowTape;
  if (yellowTape && workingWorkspace && hasManual) {
    const [enriched] = enrichDetectionsWithWorkspace([yellowTape], workingWorkspace);
    enrichedYellow = enriched ?? yellowTape;
  }

  const visibleDetections = enrichedYellow ? [...detections, enrichedYellow] : detections;

  return { detections: visibleDetections, workspace: workingWorkspace, gripperTip };
}

/** @deprecated Use detectObjectsLocal — kept for callers that only need dark blobs. */
export function detectDarkObjectsLocal(
  image: HTMLImageElement,
  _className = 'block',
  maxAnalysisSize = 320,
): VisionDetection[] {
  return detectObjectsLocal(image, maxAnalysisSize).detections;
}