export interface VisionDetection {
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  center_xy: [number, number];
  size_wh: [number, number];
  angle_deg: number;
  obb_xywha?: [number, number, number, number, number];
}

export interface VisionLatestResponse {
  width: number;
  height: number;
  camera_index?: number;
  model?: string;
  classes?: string[];
  detection_count: number;
  detections: VisionDetection[];
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

  return `${window.location.protocol}//${window.location.hostname}:8890`;
}
