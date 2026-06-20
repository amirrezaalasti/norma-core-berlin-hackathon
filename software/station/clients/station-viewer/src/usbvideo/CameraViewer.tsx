import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useLocalVisionDetections } from '@/hooks/useLocalVisionDetections';
import { useManualWorkspaceCalibration } from '@/hooks/useManualWorkspaceCalibration';
import { useVisionDetections } from '@/hooks/useVisionDetections';
import type { LiveCameraFrame } from './live-camera-store';
import { subscribeLiveCameraFrame } from './live-camera-store';
import {
  GRIPPER_TIP_LABEL,
  MANUAL_CALIBRATION_STEP_LABELS,
  screenToImagePoint,
} from './manual-workspace-calibration';
import { drawDetectionOverlay } from './vision-overlay';
import type { VisionLatestResponse } from './vision-types';

interface CameraViewerProps {
  sourceId: string | null | undefined;
  className?: string;
  imageClassName?: string;
  overlay?: 'none' | 'fps';
  fit?: 'contain' | 'cover';
  showDetectionOverlay?: boolean;
}

function toBlobPart(data: Uint8Array): BlobPart {
  if (data.buffer instanceof ArrayBuffer) {
    if (data.byteOffset === 0 && data.byteLength === data.buffer.byteLength) {
      return data.buffer;
    }

    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }

  return new Uint8Array(data).buffer;
}

const CameraViewer = memo(function CameraViewer({
  sourceId,
  className = '',
  imageClassName = '',
  overlay = 'fps',
  fit = 'contain',
  showDetectionOverlay = false,
}: CameraViewerProps) {
  const [fps, setFps] = useState<number>(0);
  const [hasImage, setHasImage] = useState(false);
  const [analysisImage, setAnalysisImage] = useState<HTMLImageElement | null>(null);
  const hasImageRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const displayedUrlRef = useRef<string | null>(null);
  const pendingUrlRef = useRef<string | null>(null);
  const lastFrameIndexRef = useRef<string | null>(null);
  const generationRef = useRef(0);
  const frameCount = useRef<number>(0);
  const lastFpsTime = useRef<number>(Date.now());
  const fitRef = useRef(fit);
  const showDetectionOverlayRef = useRef(showDetectionOverlay);
  const visionPayloadRef = useRef<VisionLatestResponse | null>(null);
  const visionErrorRef = useRef<string | null>(null);
  const pendingPointsRef = useRef<[number, number][]>([]);
  const calibrationModeRef = useRef<'idle' | 'corners' | 'gripper'>('idle');

  fitRef.current = fit;
  showDetectionOverlayRef.current = showDetectionOverlay;

  const {
    manualWorkspace,
    calibrationMode,
    pendingPoints,
    nextStepLabel,
    readyForPick,
    startCornerCalibration,
    startGripperCalibration,
    cancelCalibration,
    clearCalibration,
    addCalibrationPoint,
  } = useManualWorkspaceCalibration(sourceId);

  calibrationModeRef.current = calibrationMode;
  pendingPointsRef.current = pendingPoints;
  const isCalibrating = calibrationMode !== 'idle';

  const {
    payload: remotePayload,
    connected: remoteConnected,
    error: remoteError,
  } = useVisionDetections(showDetectionOverlay);

  const { payload: localPayload, error: localError } = useLocalVisionDetections(
    showDetectionOverlay,
    analysisImage,
    hasImage,
    manualWorkspace,
  );

  const visionPayload = localPayload
    ? {
        ...localPayload,
        workspace: manualWorkspace ?? localPayload.workspace ?? remotePayload?.workspace ?? null,
      }
    : remoteConnected && remotePayload
      ? remotePayload
      : localPayload;
  const visionError = localError ?? remoteError;

  visionPayloadRef.current = visionPayload;
  visionErrorRef.current = visionError;

  const revokeUrl = (url: string | null) => {
    if (url) {
      URL.revokeObjectURL(url);
    }
  };

  const redrawOverlay = useCallback(() => {
    if (!showDetectionOverlayRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const image = imageRef.current;
    const payload = visionPayloadRef.current;
    if (!canvas || !image || !hasImageRef.current || image.naturalWidth <= 0) {
      return;
    }

    const pendingLabels =
      calibrationModeRef.current === 'gripper'
        ? [GRIPPER_TIP_LABEL]
        : MANUAL_CALIBRATION_STEP_LABELS;

    drawDetectionOverlay(
      canvas,
      image,
      payload?.detections ?? [],
      fitRef.current,
      payload?.width ?? image.naturalWidth,
      payload?.height ?? image.naturalHeight,
      payload?.inference_fps,
      visionErrorRef.current ?? payload?.error ?? null,
      payload?.model ?? null,
      payload?.workspace ?? null,
      pendingPointsRef.current,
      pendingLabels,
    );
  }, []);

  const clearImage = useCallback((updateState = true) => {
    generationRef.current++;
    const img = imageRef.current;
    if (img) {
      img.onload = null;
      img.onerror = null;
      img.removeAttribute('src');
    }

    revokeUrl(pendingUrlRef.current);
    if (displayedUrlRef.current !== pendingUrlRef.current) {
      revokeUrl(displayedUrlRef.current);
    }
    pendingUrlRef.current = null;
    displayedUrlRef.current = null;
    lastFrameIndexRef.current = null;
    hasImageRef.current = false;
    frameCount.current = 0;
    lastFpsTime.current = Date.now();
    setAnalysisImage(null);
    if (updateState) {
      setFps(0);
      setHasImage(false);
    }
  }, []);

  const updateImage = useCallback((frame: LiveCameraFrame) => {
    if (!frame.data || frame.data.length === 0) {
      return;
    }
    if (frame.index && frame.index === lastFrameIndexRef.current) {
      return;
    }

    const img = imageRef.current;
    if (!img) {
      return;
    }

    frameCount.current++;
    const nowFps = Date.now();
    const timeDiff = nowFps - lastFpsTime.current;

    if (timeDiff >= 1000) {
      const calculatedFps = (frameCount.current / timeDiff) * 1000;
      setFps(calculatedFps);
      frameCount.current = 0;
      lastFpsTime.current = nowFps;
    }

    const url = URL.createObjectURL(new Blob([toBlobPart(frame.data)], { type: 'image/jpeg' }));
    const previousPendingUrl = pendingUrlRef.current;
    const generation = generationRef.current;

    pendingUrlRef.current = url;
    if (previousPendingUrl && previousPendingUrl !== displayedUrlRef.current) {
      URL.revokeObjectURL(previousPendingUrl);
    }

    img.onload = () => {
      if (generationRef.current !== generation || pendingUrlRef.current !== url) {
        URL.revokeObjectURL(url);
        return;
      }

      const previousDisplayedUrl = displayedUrlRef.current;
      displayedUrlRef.current = url;
      pendingUrlRef.current = null;
      lastFrameIndexRef.current = frame.index;
      if (!hasImageRef.current) {
        hasImageRef.current = true;
        setHasImage(true);
      }
      setAnalysisImage(img);

      if (previousDisplayedUrl && previousDisplayedUrl !== url) {
        URL.revokeObjectURL(previousDisplayedUrl);
      }

      requestAnimationFrame(() => redrawOverlay());
    };

    img.onerror = () => {
      if (pendingUrlRef.current === url) {
        pendingUrlRef.current = null;
      }
      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [redrawOverlay]);

  useEffect(() => {
    clearImage();
    if (!sourceId) {
      return () => clearImage(false);
    }

    const unsubscribe = subscribeLiveCameraFrame(sourceId, updateImage);
    return () => {
      unsubscribe();
      clearImage(false);
    };
  }, [clearImage, sourceId, updateImage]);

  useEffect(() => {
    redrawOverlay();
  }, [
    redrawOverlay,
    visionPayload,
    visionError,
    showDetectionOverlay,
    calibrationMode,
    pendingPoints,
    manualWorkspace,
  ]);

  const handleCalibrationClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (calibrationModeRef.current === 'idle' || !containerRef.current || !imageRef.current) {
        return;
      }

      const image = imageRef.current;
      const container = containerRef.current;
      const point = screenToImagePoint(
        event.clientX,
        event.clientY,
        container.getBoundingClientRect(),
        container.clientWidth,
        container.clientHeight,
        image.naturalWidth,
        image.naturalHeight,
        fitRef.current,
      );
      if (point) {
        addCalibrationPoint(point);
      }
    },
    [addCalibrationPoint],
  );

  useEffect(() => {
    if (!showDetectionOverlay) {
      return;
    }

    const container = containerRef.current;
    if (!container) {
      return;
    }

    const observer = new ResizeObserver(() => {
      redrawOverlay();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [redrawOverlay, showDetectionOverlay]);

  if (!sourceId) {
    return <div className="text-text-primary p-4">Waiting for USB Video data...</div>;
  }

  const fitClassName = fit === 'cover' ? 'object-cover' : 'object-contain';
  const hasCorners = Boolean(manualWorkspace?.corners_xy);

  return (
    <div className={`overflow-hidden h-full ${className}`}>
      <div
        ref={containerRef}
        className="relative flex justify-center items-center h-full w-full bg-black/20"
      >
        <img
          ref={imageRef}
          alt="USB Camera Feed"
          className={`h-full w-full ${fitClassName} ${imageClassName} ${hasImage ? '' : 'hidden'}`}
        />
        <canvas
          ref={canvasRef}
          onClick={handleCalibrationClick}
          className={`absolute inset-0 h-full w-full ${
            showDetectionOverlay && hasImage
              ? isCalibrating
                ? 'cursor-crosshair pointer-events-auto'
                : 'pointer-events-none'
              : 'hidden'
          }`}
        />
        {showDetectionOverlay && hasImage && (
          <div className="absolute bottom-2 left-2 z-40 flex flex-wrap items-center gap-2 pointer-events-auto">
            {isCalibrating ? (
              <>
                <span className="rounded bg-surface-secondary/90 px-2 py-1 text-xs font-mono text-accent-data">
                  {calibrationMode === 'gripper'
                    ? `Click ${GRIPPER_TIP_LABEL}`
                    : `Click corner ${nextStepLabel ?? '…'} (${pendingPoints.length + 1}/4)`}
                </span>
                <button
                  type="button"
                  onClick={cancelCalibration}
                  className="rounded bg-surface-secondary/90 px-2 py-1 text-xs text-text-primary hover:bg-surface-primary"
                >
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={startCornerCalibration}
                  className="rounded bg-accent-data/90 px-2 py-1 text-xs font-medium text-surface-base hover:bg-accent-data"
                >
                  Set 4 points
                </button>
                {hasCorners && (
                  <button
                    type="button"
                    onClick={startGripperCalibration}
                    className={`rounded px-2 py-1 text-xs font-medium ${
                      readyForPick
                        ? 'bg-surface-secondary/90 text-text-primary hover:bg-surface-primary'
                        : 'bg-amber-500/90 text-surface-base hover:bg-amber-400'
                    }`}
                  >
                    Set gripper tip
                  </button>
                )}
                {readyForPick && (
                  <span className="rounded bg-emerald-600/90 px-2 py-1 text-xs font-mono text-white">
                    Ready for pick
                  </span>
                )}
                {manualWorkspace && (
                  <button
                    type="button"
                    onClick={clearCalibration}
                    className="rounded bg-surface-secondary/90 px-2 py-1 text-xs text-text-primary hover:bg-surface-primary"
                  >
                    Clear
                  </button>
                )}
              </>
            )}
          </div>
        )}
        {!hasImage && (
          <div className="text-text-primary p-4">Waiting for USB Video data...</div>
        )}
        {overlay === 'fps' && (
          <div className="absolute top-0 right-0 p-2 text-right bg-surface-secondary/70 rounded-bl-lg backdrop-blur-sm">
            <span className="text-xs text-text-label">FPS: </span>
            <span className="text-xs font-mono text-accent-data">{fps.toFixed(1)}</span>
          </div>
        )}
      </div>
    </div>
  );
});

export default CameraViewer;
