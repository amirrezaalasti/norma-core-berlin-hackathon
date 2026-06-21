import type { WorkspaceCalibration } from './vision-types';

export interface CameraCalibrationData {
  resolution: [number, number];
  camera_matrix: number[][];
  distortion_coefficients: number[];
  has_extrinsics?: boolean;
  board_plane_z_mm?: number;
  T_cam2world?: number[][];
}

interface IntrinsicsFile {
  resolution?: [number, number];
  aligned?: {
    camera_matrix: number[][];
    distortion_coefficients: number[];
  };
  unaligned?: {
    camera_matrix: number[][];
    distortion_coefficients: number[];
  };
  camera_matrix?: number[][];
  distortion_coefficients?: number[];
}

interface ExtrinsicsFile {
  T_cam2world?: number[][];
  reprojection_error_px?: number;
}

let cachedCalibration: CameraCalibrationData | null | undefined;

function variantFromEnv(): 'aligned' | 'unaligned' {
  const configured = import.meta.env.VITE_INTRINSICS_VARIANT as string | undefined;
  return configured === 'unaligned' ? 'unaligned' : 'aligned';
}

function buildCalibration(
  intrinsics: IntrinsicsFile,
  extrinsics: ExtrinsicsFile | null,
  variant: 'aligned' | 'unaligned',
): CameraCalibrationData | null {
  const block = intrinsics[variant] ?? intrinsics;
  if (!block?.camera_matrix || !block.distortion_coefficients) {
    return null;
  }

  return {
    resolution: intrinsics.resolution ?? [
      Math.round(block.camera_matrix[0][2] * 2),
      Math.round(block.camera_matrix[1][2] * 2),
    ],
    camera_matrix: block.camera_matrix,
    distortion_coefficients: block.distortion_coefficients,
    has_extrinsics: Boolean(extrinsics?.T_cam2world),
    board_plane_z_mm: 0,
    T_cam2world: extrinsics?.T_cam2world,
  };
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

export async function loadCameraCalibration(): Promise<CameraCalibrationData | null> {
  if (cachedCalibration !== undefined) {
    return cachedCalibration;
  }

  const variant = variantFromEnv();
  const visionBase = (import.meta.env.VITE_VISION_API as string | undefined)?.replace(/\/$/, '');

  if (visionBase) {
    const payload = await fetchJson<CameraCalibrationData>(`${visionBase}/calibration/camera`);
    if (payload?.camera_matrix) {
      cachedCalibration = payload;
      return payload;
    }
  }

  try {
    const [intrinsicsModule, extrinsicsModule] = await Promise.all([
      import('../../../../../../images/intrinsics.json'),
      import('../../../../../../images/extrinsics.json').catch(() => null),
    ]);
    const intrinsics = (intrinsicsModule.default ?? intrinsicsModule) as unknown as IntrinsicsFile;
    const extrinsics = extrinsicsModule
      ? ((extrinsicsModule.default ?? extrinsicsModule) as unknown as ExtrinsicsFile)
      : null;
    cachedCalibration = buildCalibration(intrinsics, extrinsics, variant);
    return cachedCalibration;
  } catch {
    cachedCalibration = null;
    return null;
  }
}

function distortNormalized(x: number, y: number, k: number[]): [number, number] {
  const [k1, k2, p1, p2, k3 = 0] = k;
  const r2 = x * x + y * y;
  const r4 = r2 * r2;
  const r6 = r4 * r2;
  const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  const xDistorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yDistorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [xDistorted, yDistorted];
}

export function undistortPoint(
  px: number,
  py: number,
  calibration: CameraCalibrationData,
): [number, number] {
  const k = calibration.camera_matrix;
  const fx = k[0][0];
  const fy = k[1][1];
  const cx = k[0][2];
  const cy = k[1][2];
  const dist = calibration.distortion_coefficients;

  let x = (px - cx) / fx;
  let y = (py - cy) / fy;

  for (let i = 0; i < 8; i += 1) {
    const [xd, yd] = distortNormalized(x, y, dist);
    const errorX = xd - (px - cx) / fx;
    const errorY = yd - (py - cy) / fy;
    x -= errorX;
    y -= errorY;
  }

  return [x * fx + cx, y * fy + cy];
}

export function undistortImage(
  source: CanvasImageSource,
  width: number,
  height: number,
  calibration: CameraCalibrationData,
): ImageData | null {
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = width;
  sampleCanvas.height = height;
  const sampleCtx = sampleCanvas.getContext('2d');
  if (!sampleCtx) {
    return null;
  }
  sampleCtx.drawImage(source, 0, 0, width, height);
  const sourceData = sampleCtx.getImageData(0, 0, width, height);

  const output = new ImageData(width, height);
  const step = width > 960 ? 2 : 1;

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const [srcX, srcY] = undistortPoint(x, y, calibration);
      const sx = Math.max(0, Math.min(width - 1, Math.round(srcX)));
      const sy = Math.max(0, Math.min(height - 1, Math.round(srcY)));
      const srcIndex = (sy * width + sx) * 4;
      for (let dy = 0; dy < step && y + dy < height; dy += 1) {
        for (let dx = 0; dx < step && x + dx < width; dx += 1) {
          const dstIndex = ((y + dy) * width + (x + dx)) * 4;
          output.data[dstIndex] = sourceData.data[srcIndex];
          output.data[dstIndex + 1] = sourceData.data[srcIndex + 1];
          output.data[dstIndex + 2] = sourceData.data[srcIndex + 2];
          output.data[dstIndex + 3] = 255;
        }
      }
    }
  }

  return output;
}

export function pixelToPlaneMm(
  px: number,
  py: number,
  calibration: CameraCalibrationData,
): [number, number] | null {
  if (!calibration.T_cam2world) {
    return null;
  }

  const k = calibration.camera_matrix;
  const fx = k[0][0];
  const fy = k[1][1];
  const cx = k[0][2];
  const cy = k[1][2];
  const [ux, uy] = undistortPoint(px, py, calibration);
  const rayCam = [ (ux - cx) / fx, (uy - cy) / fy, 1 ] as const;
  const norm = Math.hypot(rayCam[0], rayCam[1], rayCam[2]);
  const directionCam = [rayCam[0] / norm, rayCam[1] / norm, rayCam[2] / norm] as const;

  const transform = calibration.T_cam2world;
  const origin = [
    transform[0][3],
    transform[1][3],
    transform[2][3],
  ] as const;
  const directionWorld = [
    transform[0][0] * directionCam[0] + transform[0][1] * directionCam[1] + transform[0][2] * directionCam[2],
    transform[1][0] * directionCam[0] + transform[1][1] * directionCam[1] + transform[1][2] * directionCam[2],
    transform[2][0] * directionCam[0] + transform[2][1] * directionCam[1] + transform[2][2] * directionCam[2],
  ] as const;

  const planeZ = calibration.board_plane_z_mm ?? 0;
  if (Math.abs(directionWorld[2]) < 1e-9) {
    return null;
  }
  const scale = (planeZ - origin[2]) / directionWorld[2];
  if (scale < 0) {
    return null;
  }

  return [
    origin[0] + scale * directionWorld[0],
    origin[1] + scale * directionWorld[1],
  ];
}

export function pixelOffsetMm(
  px: number,
  py: number,
  originPx: number,
  originPy: number,
  calibration: CameraCalibrationData,
): { offset: [number, number]; distance: number } | null {
  const obj = pixelToPlaneMm(px, py, calibration);
  const origin = pixelToPlaneMm(originPx, originPy, calibration);
  if (!obj || !origin) {
    return null;
  }
  const dx = obj[0] - origin[0];
  const dy = obj[1] - origin[1];
  return {
    offset: [dx, dy],
    distance: Math.hypot(dx, dy),
  };
}

export function withCameraWorkspaceUnits(
  workspace: WorkspaceCalibration | null,
  calibration: CameraCalibrationData | null,
): WorkspaceCalibration | null {
  if (
    !workspace ||
    !calibration?.has_extrinsics ||
    workspace.calibration_source === 'manual' ||
    workspace.calibration_source === 'apriltag' ||
    workspace.calibration_source === 'markers' ||
    workspace.calibration_source === 'blue_dots'
  ) {
    return workspace;
  }
  return {
    ...workspace,
    calibration_source: 'camera',
    units: 'mm',
    gripper_tip_set: workspace.gripper_tip_set ?? Boolean(workspace.origin_xy),
  };
}
