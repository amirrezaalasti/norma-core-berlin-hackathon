import type { VisionDetection } from './vision-types';

export const LOCAL_VISION_CLASSES = ['black block', 'red bull'] as const;

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

function isRedPixel(r: number, g: number, b: number): boolean {
  return r > 110 && r > g * 1.35 && r > b * 1.25 && saturation(r, g, b) > 0.3;
}

function isBluePixel(r: number, g: number, b: number): boolean {
  return b > 90 && b > r * 1.15 && b > g * 1.05 && saturation(r, g, b) > 0.25;
}

function isDarkPixel(r: number, g: number, b: number): boolean {
  return luminance(r, g, b) < 80 && saturation(r, g, b) < 0.3;
}

function isTablePixel(r: number, g: number, b: number): boolean {
  const lum = luminance(r, g, b);
  return lum > 190 && saturation(r, g, b) < 0.12;
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
  redRatio: number;
  blueRatio: number;
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

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ];
        for (const [nx, ny] of neighbors) {
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
  let redCount = 0;
  let blueCount = 0;
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
      if (isRedPixel(r, g, b)) {
        redCount += 1;
      }
      if (isBluePixel(r, g, b)) {
        blueCount += 1;
      }
      if (isDarkPixel(r, g, b)) {
        darkCount += 1;
      }
    }
  }

  return {
    redRatio: sampleCount > 0 ? redCount / sampleCount : 0,
    blueRatio: sampleCount > 0 ? blueCount / sampleCount : 0,
    darkRatio: sampleCount > 0 ? darkCount / sampleCount : 0,
    meanLuminance: sampleCount > 0 ? luminanceSum / sampleCount : 255,
    sampleCount,
  };
}

function hasRedBullBranding(stats: RegionStats): boolean {
  if (stats.redRatio < 0.025 || stats.blueRatio < 0.025) {
    return false;
  }
  if (stats.darkRatio > 0.5 && stats.meanLuminance < 70) {
    return false;
  }
  return stats.redRatio + stats.blueRatio >= 0.08;
}

function isBlackBlock(stats: RegionStats): boolean {
  if (stats.darkRatio >= 0.42 && stats.meanLuminance < 85) {
    return true;
  }
  if (stats.meanLuminance < 65) {
    return true;
  }
  return false;
}

function growCanFromBrandBlob(
  seed: BlobBox,
  imageData: ImageData,
  gray: Uint8Array,
  brandMask: Uint8Array,
  width: number,
  height: number,
): BlobBox | null {
  const seedCx = (seed.minX + seed.maxX) / 2;
  const seedCy = (seed.minY + seed.maxY) / 2;
  const maxRadius = Math.max(width, height) * 0.14;
  const maxRadiusSq = maxRadius * maxRadius;

  const visited = new Uint8Array(width * height);
  const stack: number[] = [];
  const blob: BlobBox = {
    minX: seed.minX,
    minY: seed.minY,
    maxX: seed.maxX,
    maxY: seed.maxY,
    count: 0,
  };

  for (let y = seed.minY; y <= seed.maxY; y += 1) {
    for (let x = seed.minX; x <= seed.maxX; x += 1) {
      const index = y * width + x;
      if (!brandMask[index]) {
        continue;
      }
      visited[index] = 1;
      stack.push(index);
    }
  }

  while (stack.length > 0) {
    const current = stack.pop();
    if (current == null) {
      continue;
    }

    const cy = Math.floor(current / width);
    const cx = current - cy * width;
    const dx = cx - seedCx;
    const dy = cy - seedCy;
    if (dx * dx + dy * dy > maxRadiusSq) {
      continue;
    }

    blob.count += 1;
    blob.minX = Math.min(blob.minX, cx);
    blob.minY = Math.min(blob.minY, cy);
    blob.maxX = Math.max(blob.maxX, cx);
    blob.maxY = Math.max(blob.maxY, cy);

    const neighbors = [
      [cx - 1, cy],
      [cx + 1, cy],
      [cx, cy - 1],
      [cx, cy + 1],
    ];
    for (const [nx, ny] of neighbors) {
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
        continue;
      }
      const neighborIndex = ny * width + nx;
      if (visited[neighborIndex]) {
        continue;
      }

      const offset = neighborIndex * 4;
      const r = imageData.data[offset];
      const g = imageData.data[offset + 1];
      const b = imageData.data[offset + 2];
      if (isTablePixel(r, g, b)) {
        continue;
      }

      const lum = gray[neighborIndex];
      const isBrand = brandMask[neighborIndex] === 1;
      const isCanBody = lum < 185 && lum > 28;
      if (!isBrand && !isCanBody) {
        continue;
      }

      visited[neighborIndex] = 1;
      stack.push(neighborIndex);
    }
  }

  const stats = sampleRegionStats(imageData, width, blob);
  if (!hasRedBullBranding(stats)) {
    return null;
  }

  return blob;
}

function blobToDetection(
  blob: BlobBox,
  className: string,
  confidence: number,
  scale: number,
): VisionDetection {
  const boxWidth = blob.maxX - blob.minX + 1;
  const boxHeight = blob.maxY - blob.minY + 1;
  const invScale = 1 / scale;
  const scaledCx = ((blob.minX + blob.maxX + 1) / 2) * invScale;
  const scaledCy = ((blob.minY + blob.maxY + 1) / 2) * invScale;

  return {
    class_name: className,
    confidence,
    bbox_xyxy: [
      blob.minX * invScale,
      blob.minY * invScale,
      (blob.maxX + 1) * invScale,
      (blob.maxY + 1) * invScale,
    ],
    center_xy: [scaledCx, scaledCy],
    size_wh: [boxWidth * invScale, boxHeight * invScale],
    angle_deg: boxWidth >= boxHeight ? 0 : 90,
  };
}

function blobDistance(blob: BlobBox, centerX: number, centerY: number): number {
  const cx = (blob.minX + blob.maxX + 1) / 2;
  const cy = (blob.minY + blob.maxY + 1) / 2;
  return (cx - centerX) ** 2 + (cy - centerY) ** 2;
}

function isValidBlobSize(blob: BlobBox, width: number, height: number): boolean {
  const area = blob.count;
  const boxWidth = blob.maxX - blob.minX + 1;
  const boxHeight = blob.maxY - blob.minY + 1;
  const aspect = Math.max(boxWidth, boxHeight) / Math.max(Math.min(boxWidth, boxHeight), 1);
  const fillRatio = area / Math.max(boxWidth * boxHeight, 1);
  const imageArea = width * height;

  return (
    area >= imageArea * 0.002 &&
    area <= imageArea * 0.35 &&
    aspect <= 6 &&
    fillRatio >= 0.25
  );
}

function bboxIoU(a: [number, number, number, number], b: [number, number, number, number]): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  if (x2 <= x1 || y2 <= y1) {
    return 0;
  }

  const intersection = (x2 - x1) * (y2 - y1);
  const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
  const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
  return intersection / (areaA + areaB - intersection);
}

function dedupeDetections(detections: VisionDetection[]): VisionDetection[] {
  const kept: VisionDetection[] = [];

  for (const detection of detections.sort((a, b) => b.confidence - a.confidence)) {
    const overlaps = kept.some(
      (existing) =>
        existing.class_name === detection.class_name &&
        bboxIoU(existing.bbox_xyxy, detection.bbox_xyxy) > 0.3,
    );
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
    .filter(({ stats }) => isBlackBlock(stats) && !hasRedBullBranding(stats))
    .map(({ blob, stats }) => ({
      detection: blobToDetection(
        blob,
        'black block',
        Math.min(0.97, 0.6 + stats.darkRatio * 0.35),
        scale,
      ),
      distance: blobDistance(blob, centerX, centerY),
    }))
    .sort((a, b) => a.distance - b.distance)
    .map((item) => item.detection);
}

function detectRedBullCans(
  brandMask: Uint8Array,
  imageData: ImageData,
  gray: Uint8Array,
  width: number,
  height: number,
  scale: number,
  centerX: number,
  centerY: number,
): VisionDetection[] {
  return findBlobs(brandMask, width, height)
    .map((seed) => growCanFromBrandBlob(seed, imageData, gray, brandMask, width, height))
    .filter((blob): blob is BlobBox => blob != null && isValidBlobSize(blob, width, height))
    .map((blob) => {
      const stats = sampleRegionStats(imageData, width, blob);
      return {
        detection: blobToDetection(
          blob,
          'red bull',
          Math.min(0.97, 0.65 + (stats.redRatio + stats.blueRatio) * 1.5),
          scale,
        ),
        distance: blobDistance(blob, centerX, centerY),
      };
    })
    .sort((a, b) => a.distance - b.distance)
    .map((item) => item.detection);
}

export function detectObjectsLocal(
  image: HTMLImageElement,
  maxAnalysisSize = 320,
): VisionDetection[] {
  if (image.naturalWidth <= 0 || image.naturalHeight <= 0) {
    return [];
  }

  const scale = Math.min(
    1,
    maxAnalysisSize / Math.max(image.naturalWidth, image.naturalHeight),
  );
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return [];
  }

  ctx.drawImage(image, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  const gray = new Uint8Array(width * height);
  const darkMask = new Uint8Array(width * height);
  const brandMask = new Uint8Array(width * height);

  for (let i = 0; i < width * height; i += 1) {
    const offset = i * 4;
    const r = imageData.data[offset];
    const g = imageData.data[offset + 1];
    const b = imageData.data[offset + 2];
    gray[i] = luminance(r, g, b);
    brandMask[i] = isRedPixel(r, g, b) || isBluePixel(r, g, b) ? 1 : 0;
  }

  const threshold = Math.min(otsuThreshold(gray) * 0.85, 120);
  for (let i = 0; i < gray.length; i += 1) {
    darkMask[i] = gray[i] < threshold ? 1 : 0;
  }

  const centerX = width / 2;
  const centerY = height / 2;

  const detections = dedupeDetections([
    ...detectRedBullCans(brandMask, imageData, gray, width, height, scale, centerX, centerY),
    ...detectBlackBlocks(darkMask, imageData, width, height, scale, centerX, centerY),
  ]).filter((detection, _, all) => {
    if (detection.class_name !== 'black block') {
      return true;
    }
    return !all.some(
      (other) =>
        other.class_name === 'red bull' &&
        bboxIoU(other.bbox_xyxy, detection.bbox_xyxy) > 0.15,
    );
  });

  return detections.slice(0, 4);
}

/** @deprecated Use detectObjectsLocal — kept for callers that only need dark blobs. */
export function detectDarkObjectsLocal(
  image: HTMLImageElement,
  _className = 'black block',
  maxAnalysisSize = 320,
): VisionDetection[] {
  return detectObjectsLocal(image, maxAnalysisSize);
}
