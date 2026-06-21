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
  /** Offset from the board reference point (+x right, +y down). Manual cal: center of 4 corners. */
  offset_xy?: [number, number];
  /** Euclidean distance from the board reference point (units follow workspace.units). */
  distance?: number;
  /** 1-indexed grid square on the manual workspace board (row-major from top-left). */
  square_id?: number;
  square_col?: number;
  square_row?: number;
  /** Normalized board position of the containing square's center. */
  square_center_board_xy?: [number, number];
  /** Offset from square center in cell units (−0.5…0.5 at cell edges). */
  square_local_xy?: [number, number];
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

/** Gripper tip position derived from manual corner homography or Roboflow detection. */
export interface GripperTipPosition {
  pixel_xy: [number, number];
  board_xy: [number, number] | null;
  offset_xy: [number, number];
  distance: number;
  class_name?: string;
  confidence?: number;
  source?: 'manual' | 'roboflow' | 'detected';
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
