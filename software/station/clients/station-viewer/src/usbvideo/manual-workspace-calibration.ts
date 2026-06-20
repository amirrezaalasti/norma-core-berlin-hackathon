import type { WorkspaceCalibration } from './vision-types';
import { getVisionApiBase } from './vision-types';

const STORAGE_PREFIX = 'norma-manual-workspace-v1';
const BOARD_WIDTH_MM = Number(import.meta.env.VITE_BOARD_WIDTH_MM ?? 280);
const BOARD_HEIGHT_MM = Number(import.meta.env.VITE_BOARD_HEIGHT_MM ?? 200);
const TAG_INSET_MM = Number(import.meta.env.VITE_TAG_INSET_MM ?? 25);

export type ManualWorkspacePayload = WorkspaceCalibration & {
  gripper_tip_set?: boolean;
};

/** Default board corners (TL, TR, BR, BL) in image pixels — hand-calibrated for this station. */
export const DEFAULT_MANUAL_WORKSPACE_CORNERS: [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
] = [
  [21.415977961432507, 181.1900826446281],
  [60.129476584022036, 36.22038567493114],
  [242.16528925619832, 34.1611570247934],
  [270.58264462809916, 170.0702479338843],
];

export const DEFAULT_MANUAL_WORKSPACE: ManualWorkspacePayload = {
  corners_xy: DEFAULT_MANUAL_WORKSPACE_CORNERS,
  center_xy: [148.573347107438, 105.41046831955924],
  width_px: 144.4490172070461,
  height_px: 215.7310656528827,
  angle_deg: -75.04832264607566,
  confidence: 1,
  origin_xy: [180.38842975206612, 137.53443526170798],
  calibration_source: 'manual',
  units: 'mm',
  plane_width: BOARD_WIDTH_MM,
  plane_height: BOARD_HEIGHT_MM,
  tag_inset_mm: TAG_INSET_MM,
  gripper_tip_set: true,
};

export const MANUAL_CALIBRATION_STEP_LABELS = ['TL', 'TR', 'BR', 'BL'] as const;
export const GRIPPER_TIP_LABEL = 'gripper tip';

function pointDistance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function buildManualWorkspace(
  points: [[number, number], [number, number], [number, number], [number, number]],
  gripperTip: [number, number] | null = null,
): ManualWorkspacePayload {
  const [topLeft, topRight, bottomRight, bottomLeft] = points;
  const centerX = (topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) / 4;
  const centerY = (topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) / 4;
  const widthPx =
    (pointDistance(topLeft, topRight) + pointDistance(bottomLeft, bottomRight)) / 2;
  const heightPx =
    (pointDistance(topLeft, bottomLeft) + pointDistance(topRight, bottomRight)) / 2;
  const angleDeg =
    (Math.atan2(topRight[1] - topLeft[1], topRight[0] - topLeft[0]) * 180) / Math.PI;

  const hasGripper = gripperTip != null;

  return {
    corners_xy: points,
    center_xy: [centerX, centerY],
    width_px: widthPx,
    height_px: heightPx,
    angle_deg: angleDeg,
    confidence: 1,
    origin_xy: hasGripper ? gripperTip : [centerX, centerY],
    calibration_source: 'manual',
    units: 'mm',
    plane_width: BOARD_WIDTH_MM,
    plane_height: BOARD_HEIGHT_MM,
    tag_inset_mm: TAG_INSET_MM,
    gripper_tip_set: hasGripper,
  };
}

function storageKey(sourceId: string): string {
  return `${STORAGE_PREFIX}:${sourceId}`;
}

export function loadManualWorkspace(sourceId: string): ManualWorkspacePayload {
  try {
    const raw = localStorage.getItem(storageKey(sourceId));
    if (raw) {
      const parsed = JSON.parse(raw) as ManualWorkspacePayload;
      if (parsed.calibration_source === 'manual' && parsed.corners_xy) {
        return parsed;
      }
    }
  } catch {
    // fall through to default workspace
  }
  return DEFAULT_MANUAL_WORKSPACE;
}

export function isManualWorkspaceReady(workspace: ManualWorkspacePayload | null): boolean {
  return Boolean(workspace?.gripper_tip_set && workspace.corners_xy);
}

export async function syncManualWorkspaceToServer(
  workspace: ManualWorkspacePayload,
): Promise<boolean> {
  try {
    const response = await fetch(`${getVisionApiBase()}/calibration/manual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(workspace),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function clearManualWorkspaceOnServer(): Promise<void> {
  try {
    await fetch(`${getVisionApiBase()}/calibration/manual`, { method: 'DELETE' });
  } catch {
    // vision server optional
  }
}

export function saveManualWorkspace(sourceId: string, workspace: ManualWorkspacePayload): void {
  localStorage.setItem(storageKey(sourceId), JSON.stringify(workspace));
  void syncManualWorkspaceToServer(workspace);
}

export function clearManualWorkspace(sourceId: string): void {
  localStorage.removeItem(storageKey(sourceId));
  void clearManualWorkspaceOnServer();
}

export function screenToImagePoint(
  clientX: number,
  clientY: number,
  containerRect: DOMRect,
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
  fit: 'contain' | 'cover',
): [number, number] | null {
  if (imageWidth <= 0 || imageHeight <= 0 || containerWidth <= 0 || containerHeight <= 0) {
    return null;
  }

  const containerAspect = containerWidth / containerHeight;
  const imageAspect = imageWidth / imageHeight;

  let layoutX: number;
  let layoutY: number;
  let layoutWidth: number;
  let layoutHeight: number;
  let scale: number;

  if (fit === 'cover') {
    scale =
      containerAspect > imageAspect
        ? containerWidth / imageWidth
        : containerHeight / imageHeight;
    layoutWidth = imageWidth * scale;
    layoutHeight = imageHeight * scale;
    layoutX = (containerWidth - layoutWidth) / 2;
    layoutY = (containerHeight - layoutHeight) / 2;
  } else {
    scale =
      containerAspect > imageAspect
        ? containerHeight / imageHeight
        : containerWidth / imageWidth;
    layoutWidth = imageWidth * scale;
    layoutHeight = imageHeight * scale;
    layoutX = (containerWidth - layoutWidth) / 2;
    layoutY = (containerHeight - layoutHeight) / 2;
  }

  const localX = clientX - containerRect.left - layoutX;
  const localY = clientY - containerRect.top - layoutY;
  if (localX < 0 || localY < 0 || localX > layoutWidth || localY > layoutHeight) {
    return null;
  }

  return [localX / scale, localY / scale];
}
