import type { CameraCalibrationData } from './camera-calibration';
import { pixelOffsetMm } from './camera-calibration';
import { enrichDetectionWithSquare } from './workspace-grid';
import type { GripperTipPosition, VisionDetection, WorkspaceCalibration } from './vision-types';

interface Point {
  x: number;
  y: number;
}

interface BlobBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  count: number;
}

const BOARD_WIDTH_MM = Number(import.meta.env.VITE_BOARD_WIDTH_MM ?? 280);
const BOARD_HEIGHT_MM = Number(import.meta.env.VITE_BOARD_HEIGHT_MM ?? 200);
const TAG_INSET_MM = Number(import.meta.env.VITE_TAG_INSET_MM ?? 25);

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

function isBoardPixel(r: number, g: number, b: number): boolean {
  const lum = luminance(r, g, b);
  const sat = saturation(r, g, b);
  return lum >= 110 && lum <= 210 && sat < 0.18;
}

function isBoardBlueDotPixel(r: number, g: number, b: number): boolean {
  const sat = saturation(r, g, b);
  return b > 75 && b > r + 8 && b >= g - 8 && sat > 0.12;
}

function morphRadius(width: number, height: number): number {
  return Math.max(10, Math.round(Math.min(width, height) * 0.024));
}

function dilateMask(mask: Uint8Array, width: number, height: number, radius: number): Uint8Array {
  const output = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) {
        continue;
      }
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          output[ny * width + nx] = 1;
        }
      }
    }
  }
  return output;
}

function erodeMask(mask: Uint8Array, width: number, height: number, radius: number): Uint8Array {
  const output = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) {
        continue;
      }
      let keep = true;
      for (let dy = -radius; dy <= radius && keep; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height || !mask[ny * width + nx]) {
            keep = false;
            break;
          }
        }
      }
      if (keep) {
        output[index] = 1;
      }
    }
  }
  return output;
}

function closeMask(mask: Uint8Array, width: number, height: number): Uint8Array {
  return erodeMask(dilateMask(mask, width, height, 3), width, height, 3);
}

function openMask(mask: Uint8Array, width: number, height: number): Uint8Array {
  return dilateMask(erodeMask(mask, width, height, 2), width, height, 2);
}

function refineBoardMask(mask: Uint8Array, width: number, height: number): Uint8Array {
  const radius = morphRadius(width, height);
  const eroded = erodeMask(mask, width, height, radius);
  const restored = dilateMask(eroded, width, height, radius);
  return openMask(closeMask(restored, width, height), width, height);
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

function collectBoundaryPoints(
  mask: Uint8Array,
  blob: BlobBox,
  width: number,
): Point[] {
  const points: Point[] = [];
  for (let y = blob.minY; y <= blob.maxY; y += 1) {
    for (let x = blob.minX; x <= blob.maxX; x += 1) {
      const index = y * width + x;
      if (!mask[index]) {
        continue;
      }

      const isEdge =
        x === blob.minX ||
        x === blob.maxX ||
        y === blob.minY ||
        y === blob.maxY ||
        !mask[index - 1] ||
        !mask[index + 1] ||
        !mask[index - width] ||
        !mask[index + width];
      if (isEdge) {
        points.push({ x, y });
      }
    }
  }
  return points;
}

function cross(o: Point, a: Point, b: Point): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHull(points: Point[]): Point[] {
  if (points.length <= 3) {
    return [...points];
  }

  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const lower: Point[] = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }

  const upper: Point[] = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const point = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function pointDistance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function orderCorners(points: Point[]): [Point, Point, Point, Point] {
  const sorted = [...points].sort((a, b) => a.y - b.y);
  const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = sorted.slice(-2).sort((a, b) => a.x - b.x);
  return [top[0], top[1], bottom[1], bottom[0]];
}

function minAreaRectCorners(blob: BlobBox): Point[] {
  return [
    { x: blob.minX, y: blob.minY },
    { x: blob.maxX, y: blob.minY },
    { x: blob.maxX, y: blob.maxY },
    { x: blob.minX, y: blob.maxY },
  ];
}

function minAreaRectFromPoints(points: Point[]): Point[] {
  if (points.length === 0) {
    return [];
  }

  let bestArea = Number.POSITIVE_INFINITY;
  let bestCorners: Point[] = minAreaRectCorners({
    minX: points[0].x,
    minY: points[0].y,
    maxX: points[0].x,
    maxY: points[0].y,
    count: 1,
  });

  const hull = convexHull(points);
  const candidates = hull.length >= 3 ? hull : points;

  for (let i = 0; i < candidates.length; i += 1) {
    const p1 = candidates[i];
    const p2 = candidates[(i + 1) % candidates.length];
    const edgeAngle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    const cos = Math.cos(-edgeAngle);
    const sin = Math.sin(-edgeAngle);

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (const point of points) {
      const rx = point.x * cos - point.y * sin;
      const ry = point.x * sin + point.y * cos;
      minX = Math.min(minX, rx);
      minY = Math.min(minY, ry);
      maxX = Math.max(maxX, rx);
      maxY = Math.max(maxY, ry);
    }

    const area = (maxX - minX) * (maxY - minY);
    if (area >= bestArea) {
      continue;
    }

    bestArea = area;
    const cosBack = Math.cos(edgeAngle);
    const sinBack = Math.sin(edgeAngle);
    const rotate = (x: number, y: number): Point => ({
      x: x * cosBack - y * sinBack,
      y: x * sinBack + y * cosBack,
    });
    bestCorners = [
      rotate(minX, minY),
      rotate(maxX, minY),
      rotate(maxX, maxY),
      rotate(minX, maxY),
    ];
  }

  return bestCorners;
}

function cornersFromBlob(blob: BlobBox, mask: Uint8Array, width: number): Point[] {
  const boundary = collectBoundaryPoints(mask, blob, width);
  if (boundary.length < 8) {
    return minAreaRectCorners(blob);
  }

  const rectCorners = minAreaRectFromPoints(boundary);
  return rectCorners.length === 4 ? rectCorners : minAreaRectCorners(blob);
}

function solveLinearSystem(matrix: number[][], values: number[]): number[] | null {
  const size = values.length;
  const augmented = matrix.map((row, index) => [...row, values[index]]);

  for (let col = 0; col < size; col += 1) {
    let pivotRow = col;
    for (let row = col + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivotRow][col])) {
        pivotRow = row;
      }
    }
    if (Math.abs(augmented[pivotRow][col]) < 1e-8) {
      return null;
    }

    [augmented[col], augmented[pivotRow]] = [augmented[pivotRow], augmented[col]];
    const pivot = augmented[col][col];
    for (let j = col; j <= size; j += 1) {
      augmented[col][j] /= pivot;
    }

    for (let row = 0; row < size; row += 1) {
      if (row === col) {
        continue;
      }
      const factor = augmented[row][col];
      for (let j = col; j <= size; j += 1) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }

  return augmented.map((row) => row[size]);
}

function workspaceHomographyDst(workspace: WorkspaceCalibration): [Point, Point, Point, Point] {
  if (
    workspace.units === 'mm' &&
    workspace.plane_width != null &&
    workspace.plane_height != null &&
    workspace.tag_inset_mm != null
  ) {
    const inset = workspace.tag_inset_mm;
    const boardW = workspace.plane_width;
    const boardH = workspace.plane_height;
    return [
      { x: inset, y: inset },
      { x: boardW - inset, y: inset },
      { x: boardW - inset, y: boardH - inset },
      { x: inset, y: boardH - inset },
    ];
  }
  return [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ];
}

function applyHomography(homography: number[], px: number, py: number): [number, number] | null {
  const [h0, h1, h2, h3, h4, h5, h6, h7] = homography;
  const denominator = h6 * px + h7 * py + 1;
  if (Math.abs(denominator) < 1e-8) {
    return null;
  }
  return [
    (h0 * px + h1 * py + h2) / denominator,
    (h3 * px + h4 * py + h5) / denominator,
  ];
}

function computeHomography(
  src: [Point, Point, Point, Point],
  dst: [Point, Point, Point, Point],
): number[] | null {
  const matrix: number[][] = [];
  const values: number[] = [];

  for (let i = 0; i < 4; i += 1) {
    const from = src[i];
    const to = dst[i];
    matrix.push([from.x, from.y, 1, 0, 0, 0, -to.x * from.x, -to.x * from.y]);
    values.push(to.x);
    matrix.push([0, 0, 0, from.x, from.y, 1, -to.y * from.x, -to.y * from.y]);
    values.push(to.y);
  }

  return solveLinearSystem(matrix, values);
}


export function workspaceOffsetScale(workspace: WorkspaceCalibration): { x: number; y: number } {
  if (
    workspace.units === 'mm' &&
    workspace.plane_width != null &&
    workspace.plane_height != null &&
    workspace.tag_inset_mm != null
  ) {
    return {
      x: Math.max(workspace.plane_width - 2 * workspace.tag_inset_mm, 1),
      y: Math.max(workspace.plane_height - 2 * workspace.tag_inset_mm, 1),
    };
  }
  return { x: workspace.plane_width ?? workspace.width_px, y: workspace.plane_height ?? workspace.height_px };
}

export function boardReferencePixel(workspace: WorkspaceCalibration): [number, number] {
  if (workspace.calibration_source === 'manual') {
    return workspace.center_xy;
  }
  return workspace.origin_xy ?? workspace.center_xy;
}

export function pixelToBoardOffset(
  px: number,
  py: number,
  workspace: WorkspaceCalibration,
): { offset_xy: [number, number]; distance: number } | null {
  const board_xy = pixelToBoardNormalized(px, py, workspace);
  if (!board_xy) {
    return null;
  }

  const scale = workspaceOffsetScale(workspace);
  const origin = boardReferencePixel(workspace);
  const origin_board = pixelToBoardNormalized(origin[0], origin[1], workspace);
  if (!origin_board) {
    return null;
  }

  const dx = (board_xy[0] - origin_board[0]) * scale.x;
  const dy = (board_xy[1] - origin_board[1]) * scale.y;
  return {
    offset_xy: [dx, dy],
    distance: Math.hypot(dx, dy),
  };
}

export function gripperTipFromPixel(
  px: number,
  py: number,
  workspace: WorkspaceCalibration,
  source: GripperTipPosition['source'] = 'detected',
  confidence?: number,
): GripperTipPosition {
  const board_xy = pixelToBoardNormalized(px, py, workspace);
  const offset = pixelToBoardOffset(px, py, workspace);
  return {
    pixel_xy: [px, py],
    board_xy,
    offset_xy: offset?.offset_xy ?? [0, 0],
    distance: offset?.distance ?? 0,
    source,
    confidence,
  };
}

export function gripperTipFromManualWorkspace(
  workspace: WorkspaceCalibration,
): GripperTipPosition | null {
  if (workspace.calibration_source !== 'manual' || workspace.gripper_tip_set !== true) {
    return null;
  }
  const origin = workspace.origin_xy ?? workspace.center_xy;
  if (!origin) {
    return null;
  }
  const board_xy = pixelToBoardNormalized(origin[0], origin[1], workspace);
  const offset = pixelToBoardOffset(origin[0], origin[1], workspace);
  return {
    pixel_xy: [origin[0], origin[1]],
    board_xy,
    offset_xy: offset?.offset_xy ?? [0, 0],
    distance: offset?.distance ?? 0,
    source: 'manual',
  };
}

export function filterDetectionsInWorkspace(
  detections: VisionDetection[],
  workspace: WorkspaceCalibration | null,
  margin = 0.05,
): VisionDetection[] {
  if (!workspace || workspace.calibration_source !== 'manual') {
    return detections;
  }
  return detections.filter((detection) => {
    const board_xy = pixelToBoardNormalized(
      detection.center_xy[0],
      detection.center_xy[1],
      workspace,
    );
    if (!board_xy) {
      return false;
    }
    const [u, v] = board_xy;
    return u >= -margin && u <= 1 + margin && v >= -margin && v <= 1 + margin;
  });
}

export function enrichDetectionsWithWorkspace(
  detections: VisionDetection[],
  workspace: WorkspaceCalibration | null,
  cameraCalibration: CameraCalibrationData | null = null,
): VisionDetection[] {
  if (!workspace) {
    return detections;
  }
  const useManualHomography = workspace.calibration_source === 'manual';
  return detections.map((detection) => {
    const board_xy = pixelToBoardNormalized(
      detection.center_xy[0],
      detection.center_xy[1],
      workspace,
    );
    const canOffset =
      workspace.calibration_source === 'manual'
        ? Boolean(workspace.corners_xy)
        : workspace.gripper_tip_set === true;

    if (
      !useManualHomography &&
      cameraCalibration?.has_extrinsics &&
      workspace.calibration_source === 'camera' &&
      canOffset
    ) {
      const origin = workspace.origin_xy ?? workspace.center_xy;
      const cameraOffset = origin
        ? pixelOffsetMm(
            detection.center_xy[0],
            detection.center_xy[1],
            origin[0],
            origin[1],
            cameraCalibration,
          )
        : null;
      if (cameraOffset) {
        const square = board_xy ? enrichDetectionWithSquare(board_xy) : null;
        return {
          ...detection,
          ...(board_xy ? { board_xy } : {}),
          offset_xy: cameraOffset.offset,
          distance: cameraOffset.distance,
          ...(square
            ? {
                square_id: square.square_id,
                square_col: square.square_col,
                square_row: square.square_row,
                square_center_board_xy: square.square_center_board_xy,
                square_local_xy: square.square_local_xy,
              }
            : {}),
        };
      }
    }

    const offset = canOffset
      ? pixelToBoardOffset(detection.center_xy[0], detection.center_xy[1], workspace)
      : null;
    const square = board_xy ? enrichDetectionWithSquare(board_xy) : null;
    if (!board_xy && !offset && !square) {
      return detection;
    }
    return {
      ...detection,
      ...(board_xy ? { board_xy } : {}),
      ...(offset ? { offset_xy: offset.offset_xy, distance: offset.distance } : {}),
      ...(square
        ? {
            square_id: square.square_id,
            square_col: square.square_col,
            square_row: square.square_row,
            square_center_board_xy: square.square_center_board_xy,
            square_local_xy: square.square_local_xy,
          }
        : {}),
    };
  });
}

export function boardNormalizedToPixel(
  u: number,
  v: number,
  workspace: WorkspaceCalibration,
): [number, number] | null {
  const corners = workspace.corners_xy.map(([x, y]) => ({ x, y })) as [Point, Point, Point, Point];
  const dst = workspaceHomographyDst(workspace);

  let planeX: number;
  let planeY: number;
  if (workspace.units === 'mm' && workspace.tag_inset_mm != null) {
    const scale = workspaceOffsetScale(workspace);
    planeX = workspace.tag_inset_mm + u * scale.x;
    planeY = workspace.tag_inset_mm + v * scale.y;
  } else {
    planeX = u;
    planeY = v;
  }

  const inverse = computeHomography(dst, corners);
  if (!inverse) {
    return null;
  }

  const mapped = applyHomography(inverse, planeX, planeY);
  if (!mapped) {
    return null;
  }
  return mapped;
}

export function pixelToBoardNormalized(
  px: number,
  py: number,
  workspace: WorkspaceCalibration,
): [number, number] | null {
  const corners = workspace.corners_xy.map(([x, y]) => ({ x, y })) as [Point, Point, Point, Point];
  const dst = workspaceHomographyDst(workspace);

  const homography = computeHomography(corners, dst);
  if (!homography) {
    return null;
  }

  const mapped = applyHomography(homography, px, py);
  if (!mapped) {
    return null;
  }

  const [mappedX, mappedY] = mapped;

  if (workspace.units === 'mm' && workspace.tag_inset_mm != null) {
    const scale = workspaceOffsetScale(workspace);
    const normalizedU = (mappedX - workspace.tag_inset_mm) / scale.x;
    const normalizedV = (mappedY - workspace.tag_inset_mm) / scale.y;
    if (normalizedU < -0.05 || normalizedU > 1.05 || normalizedV < -0.05 || normalizedV > 1.05) {
      return null;
    }
    return [
      Math.min(1, Math.max(0, normalizedU)),
      Math.min(1, Math.max(0, normalizedV)),
    ];
  }

  if (mappedX < -0.05 || mappedX > 1.05 || mappedY < -0.05 || mappedY > 1.05) {
    return null;
  }

  return [Math.min(1, Math.max(0, mappedX)), Math.min(1, Math.max(0, mappedY))];
}

function pickFourBlueDotCorners(
  candidates: Array<{ x: number; y: number; area: number }>,
): [Point, Point, Point, Point] | null {
  if (candidates.length < 3) {
    return null;
  }

  if (candidates.length === 3) {
    const ordered = orderCorners(candidates.map((c) => ({ x: c.x, y: c.y })));
    const [p0, p1, p2] = ordered;
    const d01 = pointDistance(p0, p1);
    const d12 = pointDistance(p1, p2);
    const d20 = pointDistance(p2, p0);
    let missing: Point;
    if (d01 >= d12 && d01 >= d20) {
      missing = { x: p0.x + p1.x - p2.x, y: p0.y + p1.y - p2.y };
    } else if (d12 >= d01 && d12 >= d20) {
      missing = { x: p1.x + p2.x - p0.x, y: p1.y + p2.y - p0.y };
    } else {
      missing = { x: p2.x + p0.x - p1.x, y: p2.y + p0.y - p1.y };
    }
    return orderCorners([p0, p1, p2, missing]);
  }

  if (candidates.length === 4) {
    return orderCorners(candidates.map((c) => ({ x: c.x, y: c.y })));
  }

  let best: [Point, Point, Point, Point] | null = null;
  let bestArea = Number.NEGATIVE_INFINITY;
  const choose = (start: number, picked: number[], depth: number) => {
    if (depth === 4) {
      const points = picked.map((index) => ({
        x: candidates[index].x,
        y: candidates[index].y,
      }));
      const ordered = orderCorners(points);
      const area = Math.abs(
        (ordered[1].x - ordered[0].x) * (ordered[2].y - ordered[0].y) -
          (ordered[2].x - ordered[0].x) * (ordered[1].y - ordered[0].y),
      );
      if (area > bestArea) {
        bestArea = area;
        best = ordered;
      }
      return;
    }
    for (let i = start; i < candidates.length; i += 1) {
      choose(i + 1, [...picked, i], depth + 1);
    }
  };
  choose(0, [], 0);
  return best;
}

function detectBoardBlueDots(
  imageData: ImageData,
  width: number,
  height: number,
  scale: number,
): WorkspaceCalibration | null {
  const imageArea = width * height;
  const maxArea = Math.min(imageArea * 0.004, 600);
  const blueMask = new Uint8Array(width * height);

  for (let i = 0; i < width * height; i += 1) {
    const offset = i * 4;
    const r = imageData.data[offset];
    const g = imageData.data[offset + 1];
    const b = imageData.data[offset + 2];
    blueMask[i] = isBoardBlueDotPixel(r, g, b) ? 1 : 0;
  }

  const refined = openMask(closeMask(blueMask, width, height), width, height);
  const candidates = findBlobs(refined, width, height)
    .map((blob) => ({
      x: (blob.minX + blob.maxX + 1) / 2,
      y: (blob.minY + blob.maxY + 1) / 2,
      area: blob.count,
    }))
    .filter(
      ({ x, y, area }) =>
        area >= 10 &&
        area <= maxArea &&
        y >= height * 0.18 &&
        y <= height * 0.92 &&
        x >= width * 0.08 &&
        x <= width * 0.95,
    );

  const ordered = pickFourBlueDotCorners(candidates);
  if (!ordered) {
    return null;
  }

  const widthPx =
    (pointDistance(ordered[0], ordered[1]) + pointDistance(ordered[3], ordered[2])) / 2;
  const heightPx =
    (pointDistance(ordered[0], ordered[3]) + pointDistance(ordered[1], ordered[2])) / 2;
  const aspect = Math.max(widthPx, heightPx) / Math.max(Math.min(widthPx, heightPx), 1);
  if (aspect < 1.05 || aspect > 3.8) {
    return null;
  }

  const invScale = 1 / scale;
  const corners = [
    [ordered[0].x * invScale, ordered[0].y * invScale],
    [ordered[1].x * invScale, ordered[1].y * invScale],
    [ordered[2].x * invScale, ordered[2].y * invScale],
    [ordered[3].x * invScale, ordered[3].y * invScale],
  ] as WorkspaceCalibration['corners_xy'];

  const centerX = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
  const centerY = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
  const angleDeg =
    (Math.atan2(ordered[1].y - ordered[0].y, ordered[1].x - ordered[0].x) * 180) / Math.PI;

  return {
    corners_xy: corners,
    center_xy: [centerX, centerY],
    width_px: widthPx * invScale,
    height_px: heightPx * invScale,
    angle_deg: angleDeg,
    confidence: Math.min(0.97, 0.78 + 0.04 * candidates.length),
    origin_xy: [centerX, centerY],
    calibration_source: 'blue_dots',
    units: 'mm',
    plane_width: BOARD_WIDTH_MM,
    plane_height: BOARD_HEIGHT_MM,
    tag_inset_mm: TAG_INSET_MM,
  };
}

function validateMarkerQuad(
  ordered: [Point, Point, Point, Point],
  boardCorners: [Point, Point, Point, Point],
  width: number,
  height: number,
): boolean {
  const boardDiag = Math.hypot(boardCorners[0].x - boardCorners[2].x, boardCorners[0].y - boardCorners[2].y);
  if (boardDiag <= 1) {
    return false;
  }

  for (let i = 0; i < 4; i += 1) {
    const dist = pointDistance(ordered[i], boardCorners[i]);
    if (dist > boardDiag * 0.35) {
      return false;
    }
  }

  const margin = Math.min(width, height) * 0.02;
  const boardInset = boardCorners.every(
    (corner) => corner.x > margin && corner.y > margin && corner.x < width - margin && corner.y < height - margin,
  );
  if (boardInset) {
    for (const point of ordered) {
      if (point.x <= margin || point.y <= margin || point.x >= width - margin || point.y >= height - margin) {
        return false;
      }
    }
  }

  const widthPx =
    (pointDistance(ordered[0], ordered[1]) + pointDistance(ordered[3], ordered[2])) / 2;
  const heightPx =
    (pointDistance(ordered[0], ordered[3]) + pointDistance(ordered[1], ordered[2])) / 2;
  const aspect = Math.max(widthPx, heightPx) / Math.max(Math.min(widthPx, heightPx), 1);
  return aspect >= 1.1 && aspect <= 3.5;
}

function pickFourNearBoardCorners(
  candidates: Array<{ x: number; y: number; area: number }>,
  boardCorners: [Point, Point, Point, Point],
  width: number,
  height: number,
): [Point, Point, Point, Point] | null {
  if (candidates.length < 4) {
    return null;
  }

  const [topLeft, topRight, bottomRight, bottomLeft] = orderCorners(boardCorners);
  const boardCornersOrdered: [Point, Point, Point, Point] = [topLeft, topRight, bottomRight, bottomLeft];
  const boardWidth = pointDistance(topLeft, topRight);
  const boardHeight = pointDistance(topLeft, bottomLeft);
  const searchRadius = Math.max(boardWidth, boardHeight) * 0.32;
  const imageArea = width * height;
  const tagMinArea = Math.max(imageArea * 0.00015, 12);
  const tagMaxArea = imageArea * 0.012;
  const margin = Math.min(width, height) * 0.025;

  const selected: Point[] = [];
  for (const corner of boardCornersOrdered) {
    const inRange = candidates.filter(
      (candidate) =>
        pointDistance(candidate, corner) <= searchRadius &&
        candidate.x >= margin &&
        candidate.y >= margin &&
        candidate.x <= width - margin &&
        candidate.y <= height - margin,
    );
    if (inRange.length === 0) {
      return null;
    }

    const tagLike = inRange.filter(
      (candidate) => candidate.area >= tagMinArea && candidate.area <= tagMaxArea,
    );
    const pool = tagLike.length > 0 ? tagLike : inRange;
    const best = pool.reduce((closest, candidate) =>
      pointDistance(candidate, corner) < pointDistance(closest, corner) ? candidate : closest,
    );
    selected.push({ x: best.x, y: best.y });
  }

  const ordered = orderCorners(selected);
  if (!validateMarkerQuad(ordered, boardCornersOrdered, width, height)) {
    return null;
  }
  return ordered;
}

function detectFourCornerMarkers(
  imageData: ImageData,
  width: number,
  height: number,
  scale: number,
  board: WorkspaceCalibration,
): WorkspaceCalibration | null {
  const boardCorners = board.corners_xy.map(([x, y]) => ({
    x: x * scale,
    y: y * scale,
  })) as [Point, Point, Point, Point];

  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i += 1) {
    const offset = i * 4;
    gray[i] = luminance(
      imageData.data[offset],
      imageData.data[offset + 1],
      imageData.data[offset + 2],
    );
  }

  const darkMask = new Uint8Array(width * height);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      let sum = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          sum += gray[index + dy * width + dx];
        }
      }
      const localMean = sum / 9;
      darkMask[index] = gray[index] < localMean - 10 ? 1 : 0;
    }
  }

  const refined = openMask(closeMask(darkMask, width, height), width, height);
  const imageArea = width * height;
  const candidates = findBlobs(refined, width, height)
    .map((blob) => {
      const boxWidth = blob.maxX - blob.minX + 1;
      const boxHeight = blob.maxY - blob.minY + 1;
      const area = blob.count;
      const aspect = Math.max(boxWidth, boxHeight) / Math.max(Math.min(boxWidth, boxHeight), 1);
      const fillRatio = area / Math.max(boxWidth * boxHeight, 1);
      const cx = (blob.minX + blob.maxX + 1) / 2;
      const cy = (blob.minY + blob.maxY + 1) / 2;
      return { area, aspect, fillRatio, cx, cy };
    })
    .filter(({ area, aspect, fillRatio }) => {
      return (
        area >= Math.max(imageArea * 0.00015, 12) &&
        area <= imageArea * 0.08 &&
        aspect <= 1.8 &&
        fillRatio >= 0.4
      );
    })
    .map(({ cx, cy, area }) => ({ x: cx, y: cy, area }));

  const ordered = pickFourNearBoardCorners(candidates, boardCorners, width, height);
  if (!ordered) {
    return null;
  }

  const invScale = 1 / scale;
  const corners = [
    [ordered[0].x * invScale, ordered[0].y * invScale],
    [ordered[1].x * invScale, ordered[1].y * invScale],
    [ordered[2].x * invScale, ordered[2].y * invScale],
    [ordered[3].x * invScale, ordered[3].y * invScale],
  ] as WorkspaceCalibration['corners_xy'];

  const centerX = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
  const centerY = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;
  const widthPx =
    (pointDistance(ordered[0], ordered[1]) + pointDistance(ordered[3], ordered[2])) / 2 * invScale;
  const heightPx =
    (pointDistance(ordered[0], ordered[3]) + pointDistance(ordered[1], ordered[2])) / 2 * invScale;
  const angleDeg =
    (Math.atan2(ordered[1].y - ordered[0].y, ordered[1].x - ordered[0].x) * 180) / Math.PI;

  return {
    corners_xy: corners,
    center_xy: [centerX, centerY],
    width_px: widthPx,
    height_px: heightPx,
    angle_deg: angleDeg,
    confidence: 0.86,
    origin_xy: [centerX, centerY],
    calibration_source: 'markers',
    units: 'px',
    plane_width: widthPx,
    plane_height: heightPx,
  };
}

export function detectWorkspace(
  imageData: ImageData,
  width: number,
  height: number,
  scale: number,
  externalWorkspace: WorkspaceCalibration | null = null,
): WorkspaceCalibration | null {
  if (externalWorkspace?.calibration_source === 'apriltag') {
    return externalWorkspace;
  }

  const blueDots = detectBoardBlueDots(imageData, width, height, scale);
  if (blueDots) {
    return blueDots;
  }

  const board = detectWorkspaceBoard(imageData, width, height, scale);
  if (!board) {
    return externalWorkspace;
  }

  const markerWorkspace = detectFourCornerMarkers(imageData, width, height, scale, board);
  if (markerWorkspace) {
    return markerWorkspace;
  }

  return board;
}

function countBorderTouches(blob: BlobBox, width: number, height: number): number {
  return (
    (blob.minX <= 5 ? 1 : 0) +
    (blob.minY <= 5 ? 1 : 0) +
    (blob.maxX >= width - 6 ? 1 : 0) +
    (blob.maxY >= height - 6 ? 1 : 0)
  );
}

export function detectWorkspaceBoard(
  imageData: ImageData,
  width: number,
  height: number,
  scale: number,
): WorkspaceCalibration | null {
  const boardMask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i += 1) {
    const offset = i * 4;
    const r = imageData.data[offset];
    const g = imageData.data[offset + 1];
    const b = imageData.data[offset + 2];
    boardMask[i] = isBoardPixel(r, g, b) ? 1 : 0;
  }

  const refinedMask = refineBoardMask(boardMask, width, height);
  const imageArea = width * height;

  const blobs = findBlobs(refinedMask, width, height)
    .map((blob) => {
      const boxWidth = blob.maxX - blob.minX + 1;
      const boxHeight = blob.maxY - blob.minY + 1;
      const fillRatio = blob.count / Math.max(boxWidth * boxHeight, 1);
      const aspect = Math.max(boxWidth, boxHeight) / Math.max(Math.min(boxWidth, boxHeight), 1);
      const borderTouches = countBorderTouches(blob, width, height);
      const score = blob.count * (0.5 + fillRatio);
      return { blob, fillRatio, aspect, borderTouches, score };
    })
    .filter(({ blob, fillRatio, aspect, borderTouches }) => {
      const area = blob.count;
      return (
        area >= imageArea * 0.06 &&
        area <= imageArea * 0.55 &&
        aspect <= 2.5 &&
        fillRatio >= 0.45 &&
        borderTouches < 2
      );
    })
    .sort((a, b) => b.score - a.score);

  const candidate = blobs[0];
  if (!candidate) {
    return null;
  }

  const { blob, fillRatio } = candidate;
  const rawCorners = cornersFromBlob(blob, refinedMask, width);
  const [topLeft, topRight, bottomRight, bottomLeft] = orderCorners(rawCorners);
  const invScale = 1 / scale;

  const corners = [
    [topLeft.x * invScale, topLeft.y * invScale],
    [topRight.x * invScale, topRight.y * invScale],
    [bottomRight.x * invScale, bottomRight.y * invScale],
    [bottomLeft.x * invScale, bottomLeft.y * invScale],
  ] as WorkspaceCalibration['corners_xy'];

  const centerX =
    (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
  const centerY =
    (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;

  const widthPx =
    (pointDistance(topLeft, topRight) + pointDistance(bottomLeft, bottomRight)) / 2 * invScale;
  const heightPx =
    (pointDistance(topLeft, bottomLeft) + pointDistance(topRight, bottomRight)) / 2 * invScale;
  const angleDeg =
    (Math.atan2(topRight.y - topLeft.y, topRight.x - topLeft.x) * 180) / Math.PI;

  const origin: [number, number] = [centerX, centerY];

  return {
    corners_xy: corners,
    center_xy: [centerX, centerY],
    width_px: widthPx,
    height_px: heightPx,
    angle_deg: angleDeg,
    confidence: Math.min(0.99, 0.55 + fillRatio * 0.4),
    origin_xy: origin,
    calibration_source: 'board',
  };
}
