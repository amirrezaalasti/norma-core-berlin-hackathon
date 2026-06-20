import { useEffect, useRef, useState } from 'react';
import { detectObjectsLocal, LOCAL_VISION_CLASSES } from '@/usbvideo/local-contrast-vision';
import type { VisionDetection, VisionLatestResponse } from '@/usbvideo/vision-types';

const ANALYSIS_INTERVAL_MS = 150;

export interface UseLocalVisionDetectionsResult {
  payload: VisionLatestResponse | null;
  error: string | null;
}

export function useLocalVisionDetections(
  enabled: boolean,
  image: HTMLImageElement | null,
  hasImage: boolean,
): UseLocalVisionDetectionsResult {
  const [payload, setPayload] = useState<VisionLatestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastDetectionsRef = useRef<VisionDetection[]>([]);

  useEffect(() => {
    if (!enabled) {
      lastDetectionsRef.current = [];
      setPayload(null);
      setError(null);
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
        const detections = detectObjectsLocal(image);
        if (detections.length > 0) {
          lastDetectionsRef.current = detections;
        }

        if (cancelled) {
          return;
        }

        setPayload({
          width: image.naturalWidth,
          height: image.naturalHeight,
          model: 'local-contrast',
          classes: [...LOCAL_VISION_CLASSES],
          detection_count: lastDetectionsRef.current.length,
          detections: lastDetectionsRef.current,
          inference_fps: 1000 / ANALYSIS_INTERVAL_MS,
          updated_at_ms: Date.now(),
          error: null,
        });
        setError(null);
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
  }, [enabled, hasImage, image]);

  return { payload, error };
}
