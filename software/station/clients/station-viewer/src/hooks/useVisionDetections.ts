import { useEffect, useState } from 'react';
import { getVisionApiBase, type VisionLatestResponse } from '@/usbvideo/vision-types';

const POLL_INTERVAL_MS = 200;

export interface UseVisionDetectionsResult {
  payload: VisionLatestResponse | null;
  connected: boolean;
  error: string | null;
}

export function useVisionDetections(enabled: boolean): UseVisionDetectionsResult {
  const [payload, setPayload] = useState<VisionLatestResponse | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setPayload(null);
      setConnected(false);
      setError(null);
      return;
    }

    let cancelled = false;
    const apiBase = getVisionApiBase();

    const poll = async () => {
      try {
        const response = await fetch(`${apiBase}/latest`, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`Vision API returned ${response.status}`);
        }
        const data = (await response.json()) as VisionLatestResponse;
        if (cancelled) {
          return;
        }
        setPayload(data);
        setConnected(true);
        setError(data.error ?? null);
      } catch (err) {
        if (cancelled) {
          return;
        }
        setConnected(false);
        setError(err instanceof Error ? err.message : 'Vision API unavailable');
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [enabled]);

  return { payload, connected, error };
}
