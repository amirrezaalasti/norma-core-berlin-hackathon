import { useCallback, useEffect, useState } from 'react';
import {
  buildManualWorkspace,
  clearManualWorkspace,
  DEFAULT_MANUAL_WORKSPACE,
  GRIPPER_TIP_LABEL,
  isManualWorkspaceReady,
  loadManualWorkspace,
  MANUAL_CALIBRATION_STEP_LABELS,
  saveManualWorkspace,
  syncManualWorkspaceToServer,
  type ManualWorkspacePayload,
} from '@/usbvideo/manual-workspace-calibration';

export type CalibrationMode = 'idle' | 'corners' | 'gripper';

export interface UseManualWorkspaceCalibrationResult {
  manualWorkspace: ManualWorkspacePayload | null;
  calibrationMode: CalibrationMode;
  pendingPoints: [number, number][];
  nextStepLabel: string | null;
  readyForPick: boolean;
  startCornerCalibration: () => void;
  startGripperCalibration: () => void;
  cancelCalibration: () => void;
  clearCalibration: () => void;
  addCalibrationPoint: (point: [number, number]) => void;
}

export function useManualWorkspaceCalibration(
  sourceId: string | null | undefined,
): UseManualWorkspaceCalibrationResult {
  const [manualWorkspace, setManualWorkspace] = useState<ManualWorkspacePayload | null>(null);
  const [calibrationMode, setCalibrationMode] = useState<CalibrationMode>('idle');
  const [pendingPoints, setPendingPoints] = useState<[number, number][]>([]);

  useEffect(() => {
    if (!sourceId) {
      setManualWorkspace(null);
      return;
    }
    const workspace = loadManualWorkspace(sourceId);
    setManualWorkspace(workspace);
    void syncManualWorkspaceToServer(workspace);
  }, [sourceId]);

  const startCornerCalibration = useCallback(() => {
    setCalibrationMode('corners');
    setPendingPoints([]);
  }, []);

  const startGripperCalibration = useCallback(() => {
    setCalibrationMode('gripper');
    setPendingPoints([]);
  }, []);

  const cancelCalibration = useCallback(() => {
    setCalibrationMode('idle');
    setPendingPoints([]);
  }, []);

  const clearCalibration = useCallback(() => {
    if (sourceId) {
      clearManualWorkspace(sourceId);
    }
    setManualWorkspace(DEFAULT_MANUAL_WORKSPACE);
    void syncManualWorkspaceToServer(DEFAULT_MANUAL_WORKSPACE);
    setCalibrationMode('idle');
    setPendingPoints([]);
  }, [sourceId]);

  const addCalibrationPoint = useCallback(
    (point: [number, number]) => {
      if (calibrationMode === 'corners') {
        const next: [number, number][] = [...pendingPoints, point];
        if (next.length < 4) {
          setPendingPoints(next);
          return;
        }

        const workspace = buildManualWorkspace(
          next as [[number, number], [number, number], [number, number], [number, number]],
          manualWorkspace?.gripper_tip_set ? manualWorkspace.origin_xy ?? null : null,
        );
        if (sourceId) {
          saveManualWorkspace(sourceId, workspace);
        }
        setManualWorkspace(workspace);
        setPendingPoints([]);
        setCalibrationMode('idle');
        return;
      }

      if (calibrationMode === 'gripper') {
        if (!manualWorkspace?.corners_xy) {
          setCalibrationMode('idle');
          return;
        }

        const workspace = buildManualWorkspace(manualWorkspace.corners_xy, point);
        if (sourceId) {
          saveManualWorkspace(sourceId, workspace);
        }
        setManualWorkspace(workspace);
        setPendingPoints([]);
        setCalibrationMode('idle');
      }
    },
    [calibrationMode, manualWorkspace, pendingPoints, sourceId],
  );

  const nextStepLabel =
    calibrationMode === 'corners' && pendingPoints.length < MANUAL_CALIBRATION_STEP_LABELS.length
      ? MANUAL_CALIBRATION_STEP_LABELS[pendingPoints.length]
      : calibrationMode === 'gripper'
        ? GRIPPER_TIP_LABEL
        : null;

  return {
    manualWorkspace,
    calibrationMode,
    pendingPoints,
    nextStepLabel,
    readyForPick: isManualWorkspaceReady(manualWorkspace),
    startCornerCalibration,
    startGripperCalibration,
    cancelCalibration,
    clearCalibration,
    addCalibrationPoint,
  };
}
