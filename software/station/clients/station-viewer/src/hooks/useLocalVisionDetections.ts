import { useEffect, useRef, useState } from 'react';
import { loadCameraCalibration, type CameraCalibrationData } from '@/usbvideo/camera-calibration';
import { detectObjectsLocal, LOCAL_VISION_CLASSES } from '@/usbvideo/local-contrast-vision';
import { isManualWorkspaceReady } from '@/usbvideo/manual-workspace-calibration';
import { gripperTipFromManualWorkspace } from '@/usbvideo/workspace-detection';
import { getVisionApiBase, type VisionDetection, type VisionLatestResponse, type WorkspaceCalibration } from '@/usbvideo/vision-types';

const ANALYSIS_INTERVAL_MS = 150;
const WORKSPACE_POLL_INTERVAL_MS = 300;

function hasManualCorners(workspace: WorkspaceCalibration | null): boolean {
  return Boolean(workspace?.corners_xy && workspace.calibration_source === 'manual');
}

export interface UseLocalVisionDetectionsResult {
  payload: VisionLatestResponse | null;
  error: string | null;
}

export function useLocalVisionDetections(
  enabled: boolean,
  image: HTMLImageElement | null,
  hasImage: boolean,
  manualWorkspace: WorkspaceCalibration | null = null,
): UseLocalVisionDetectionsResult {
  const [payload, setPayload] = useState<VisionLatestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastDetectionsRef = useRef<VisionDetection[]>([]);
  const apriltagWorkspaceRef = useRef<WorkspaceCalibration | null>(null);
  const manualWorkspaceRef = useRef<WorkspaceCalibration | null>(manualWorkspace);
  const cameraCalibrationRef = useRef<CameraCalibrationData | null>(null);

  manualWorkspaceRef.current = manualWorkspace;

  useEffect(() => {
    let cancelled = false;
    void loadCameraCalibration().then((calibration) => {
      if (!cancelled) {
        cameraCalibrationRef.current = calibration;
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!enabled) {
      lastDetectionsRef.current = [];
      apriltagWorkspaceRef.current = null;
      setPayload(null);
      setError(null);
      return;
    }

    let cancelled = false;
    const apiBase = getVisionApiBase();

    const pollWorkspace = async () => {
      if (manualWorkspaceRef.current) {
        return;
      }
      try {
        const response = await fetch(`${apiBase}/latest`, { cache: 'no-store' });
        if (!response.ok) {
          return;
        }
        const data = (await response.json()) as VisionLatestResponse;
        if (cancelled) {
          return;
        }
        if (data.workspace?.calibration_source === 'apriltag') {
          apriltagWorkspaceRef.current = data.workspace;
        }
      } catch {
        // vision server optional when using manual calibration
      }
    };

    void pollWorkspace();
    const workspaceTimer = window.setInterval(() => {
      void pollWorkspace();
    }, WORKSPACE_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(workspaceTimer);
    };
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    if (!hasImage || !image) {
      return;
    }

    let cancelled = false;

    const analyze = () => {
      if (cancelled || !image.complete || image.naturalWidth <= 0) {
        return;
      }

      try {
        const manual = manualWorkspaceRef.current;
        const useManual = hasManualCorners(manual);
        const blockResult = detectObjectsLocal(
          image,
          480,
          apriltagWorkspaceRef.current,
          useManual ? manual : null,
          useManual ? null : cameraCalibrationRef.current,
        );
        const workspace =
          (useManual ? manual : null) ??
          blockResult.workspace ??
          (apriltagWorkspaceRef.current?.calibration_source === 'apriltag'
            ? apriltagWorkspaceRef.current
            : null);
        const detections = blockResult.detections;
        const gripperTip = manual && isManualWorkspaceReady(manual)
          ? gripperTipFromManualWorkspace(manual)
          : null;

        if (detections.length > 0) {
          lastDetectionsRef.current = detections;
        }

        if (cancelled) {
          return;
        }

        let visionError: string | null = null;
        if (!workspace) {
          visionError = "Calibrate workspace: click 'Set 4 points' then 'Set gripper tip'";
        } else if (manual && !isManualWorkspaceReady(manual)) {
          visionError = "Click 'Set gripper tip' on the gripper to enable pick offsets";
        }

        setPayload({
          width: image.naturalWidth,
          height: image.naturalHeight,
          model: 'local-contrast',
          classes: [...LOCAL_VISION_CLASSES],
          detection_count: lastDetectionsRef.current.length,
          detections: lastDetectionsRef.current,
          workspace,
          gripper_tip: gripperTip,
          inference_fps: 1000 / ANALYSIS_INTERVAL_MS,
          updated_at_ms: Date.now(),
          error: visionError,
        });
        setError(visionError);
      } catch (err) {
        if (cancelled) {
          return;
        }
        setError(err instanceof Error ? err.message : 'Local vision failed');
      }
    };

    analyze();
    const timer = window.setInterval(analyze, ANALYSIS_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [enabled, hasImage, image, manualWorkspace]);

  return { payload, error };
}
