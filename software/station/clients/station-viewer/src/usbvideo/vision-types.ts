export interface VisionDetection {
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  center_xy: [number, number];
  size_wh: [number, number];
  angle_deg: number;
  obb_xywha?: [number, number, number, number, number];
  /** Normalized position within the detected workspace board (0–1). */
  board_xy?: [number, number];
  /** Offset from the gripper-tip origin in board-plane pixels (+x right, +y down). */
  offset_xy?: [number, number];
  /** Euclidean distance from the gripper-tip origin (units follow workspace.units). */
  distance?: number;
}

/** White-board workspace used as the robot's 2D environment frame. */
export interface WorkspaceCalibration {
  corners_xy: [[number, number], [number, number], [number, number], [number, number]];
  center_xy: [number, number];
  /** Gripper-tip marker (blue dot) in image pixels; falls back to center_xy when absent. */
  origin_xy?: [number, number] | null;
  width_px: number;
  height_px: number;
  angle_deg: number;
  confidence: number;
  calibration_source?: 'board' | 'gripper' | 'apriltag' | 'markers' | 'blue_dots' | 'manual' | 'camera';
  units?: 'px' | 'mm';
  plane_width?: number;
  plane_height?: number;
  tag_inset_mm?: number;
  tag_ids?: number[];
  tag_family?: string;
  /** Manual calibration: gripper tip selected as pick origin. */
  gripper_tip_set?: boolean;
}

/** Gripper tip position derived from manual corner homography. */
export interface GripperTipPosition {
  pixel_xy: [number, number];
  board_xy: [number, number] | null;
  offset_xy: [number, number];
  distance: number;
}

export interface VisionLatestResponse {
  width: number;
  height: number;
  camera_index?: number;
  model?: string;
  classes?: string[];
  detection_count: number;
  detections: VisionDetection[];
  workspace?: WorkspaceCalibration | null;
  gripper_tip?: GripperTipPosition | null;
  inference_fps?: number;
  updated_at_ms?: number;
  error?: string | null;
}

export function getVisionApiBase(): string {
  const configured = import.meta.env.VITE_VISION_API as string | undefined;
  if (configured) {
    return configured.replace(/\/$/, '');
  }

  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:8890';
  }

  // Vite dev server proxies /vision -> norma-vision-live on 8890
  if (window.location.port === '5173') {
    return `${window.location.protocol}//${window.location.host}/vision`;
  }

  return `${window.location.protocol}//${window.location.hostname}:8890`;
}
