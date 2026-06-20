import type { VisionDetection, WorkspaceCalibration } from './vision-types';
import { gripperTipFromManualWorkspace } from './workspace-detection';

export interface ImageLayoutRect {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
}

export function getImageLayoutRect(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
  fit: 'contain' | 'cover',
): ImageLayoutRect {
  if (containerWidth <= 0 || containerHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return { x: 0, y: 0, width: 0, height: 0, scale: 1 };
  }

  const containerAspect = containerWidth / containerHeight;
  const imageAspect = imageWidth / imageHeight;

  if (fit === 'cover') {
    const scale =
      containerAspect > imageAspect
        ? containerWidth / imageWidth
        : containerHeight / imageHeight;
    const width = imageWidth * scale;
    const height = imageHeight * scale;
    return {
      x: (containerWidth - width) / 2,
      y: (containerHeight - height) / 2,
      width,
      height,
      scale,
    };
  }

  const scale =
    containerAspect > imageAspect
      ? containerHeight / imageHeight
      : containerWidth / imageWidth;
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: (containerWidth - width) / 2,
    y: (containerHeight - height) / 2,
    width,
    height,
    scale,
  };
}

function mapPoint(
  x: number,
  y: number,
  layout: ImageLayoutRect,
): [number, number] {
  return [layout.x + x * layout.scale, layout.y + y * layout.scale];
}

function drawObb(
  ctx: CanvasRenderingContext2D,
  detection: VisionDetection,
  layout: ImageLayoutRect,
  color: string,
) {
  const [centerX, centerY] = mapPoint(detection.center_xy[0], detection.center_xy[1], layout);
  const width = detection.size_wh[0] * layout.scale;
  const height = detection.size_wh[1] * layout.scale;
  const angleRad = (detection.angle_deg * Math.PI) / 180;

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(angleRad);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(-width / 2, -height / 2, width, height);

  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(width / 2, 0);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawWorkspace(
  ctx: CanvasRenderingContext2D,
  workspace: WorkspaceCalibration,
  layout: ImageLayoutRect,
) {
  const corners = workspace.corners_xy.map(([x, y]) => mapPoint(x, y, layout));
  const originPoint = workspace.origin_xy ?? workspace.center_xy;
  const [originX, originY] = mapPoint(originPoint[0], originPoint[1], layout);
  const isApriltag = workspace.calibration_source === 'apriltag';
  const isMarkers = workspace.calibration_source === 'markers';
  const isBlueDots = workspace.calibration_source === 'blue_dots';
  const isManual = workspace.calibration_source === 'manual';
  const isCamera = workspace.calibration_source === 'camera';
  const isFiducial = isApriltag || isMarkers || isBlueDots || isManual || isCamera;
  const isGripper = workspace.calibration_source === 'gripper';

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]);
  for (let i = 1; i < corners.length; i += 1) {
    ctx.lineTo(corners[i][0], corners[i][1]);
  }
  ctx.closePath();
  ctx.strokeStyle = isFiducial ? 'rgba(96, 165, 250, 0.98)' : 'rgba(56, 189, 248, 0.95)';
  ctx.lineWidth = isFiducial ? 3 : 2;
  ctx.setLineDash(isFiducial ? [] : [8, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  if (isFiducial) {
    const tagLabels = ['TL', 'TR', 'BR', 'BL'];
    corners.forEach(([x, y], index) => {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = isBlueDots ? '#38bdf8' : '#fbbf24';
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      const tagId = workspace.tag_ids?.[index];
      const label = tagId != null ? `#${tagId}` : tagLabels[index];
      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = isBlueDots ? '#38bdf8' : '#fbbf24';
      ctx.fillText(label, x + 6, y - 4);
    });
  }

  const [tl, tr, , bl] = workspace.corners_xy;
  const axisScale = 0.15;
  const xEnd = mapPoint(
    originPoint[0] + (tr[0] - tl[0]) * axisScale,
    originPoint[1] + (tr[1] - tl[1]) * axisScale,
    layout,
  );
  const yEnd = mapPoint(
    originPoint[0] + (bl[0] - tl[0]) * axisScale,
    originPoint[1] + (bl[1] - tl[1]) * axisScale,
    layout,
  );

  ctx.beginPath();
  ctx.moveTo(originX, originY);
  ctx.lineTo(xEnd[0], xEnd[1]);
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#ef4444';
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText('+x', xEnd[0] + 4, xEnd[1] + 4);

  ctx.beginPath();
  ctx.moveTo(originX, originY);
  ctx.lineTo(yEnd[0], yEnd[1]);
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#22c55e';
  ctx.fillText('+y', yEnd[0] + 4, yEnd[1] + 4);

  const family = workspace.tag_family ? ` ${workspace.tag_family}` : '';
  const hasGripperOrigin = isManual
    ? workspace.gripper_tip_set === true
    : isApriltag || isBlueDots || isMarkers || isGripper;

  ctx.beginPath();
  ctx.arc(originX, originY, 7, 0, Math.PI * 2);
  ctx.fillStyle = isManual && !hasGripperOrigin ? '#f59e0b' : isFiducial ? '#2563eb' : isGripper ? '#38bdf8' : '#3b82f6';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();

  if (isManual && hasGripperOrigin) {
    const tip = gripperTipFromManualWorkspace(workspace);
    if (tip?.board_xy) {
      const unitLabel = workspace.units === 'mm' ? 'mm' : 'px';
      const tipLabel = `tip (${tip.board_xy[0].toFixed(2)}, ${tip.board_xy[1].toFixed(2)}) | x:0.0 y:0.0 d:0.0${unitLabel}`;
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      const tipTextWidth = ctx.measureText(tipLabel).width;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
      ctx.fillRect(originX + 10, originY - 10, tipTextWidth + 8, 16);
      ctx.fillStyle = '#60a5fa';
      ctx.fillText(tipLabel, originX + 14, originY + 2);
    }
  }

  const label = isManual
    ? hasGripperOrigin
      ? `manual + gripper tip ${(workspace.confidence * 100).toFixed(0)}%`
      : `manual 4-point (set gripper tip) ${(workspace.confidence * 100).toFixed(0)}%`
    : isApriltag
      ? `apriltag origin${family} ${(workspace.confidence * 100).toFixed(0)}%`
      : isBlueDots
        ? `4 blue dots ${(workspace.confidence * 100).toFixed(0)}%`
        : isMarkers
          ? `4-marker homography ${(workspace.confidence * 100).toFixed(0)}%`
          : isCamera
            ? `camera cal ${(workspace.confidence * 100).toFixed(0)}%`
            : isGripper
            ? `blue dot origin ${(workspace.confidence * 100).toFixed(0)}%`
            : `board center ${(workspace.confidence * 100).toFixed(0)}%`;
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  const textWidth = ctx.measureText(label).width;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
  ctx.fillRect(corners[0][0], corners[0][1] - 18, textWidth + 8, 18);
  ctx.fillStyle = '#38bdf8';
  ctx.fillText(label, corners[0][0] + 4, corners[0][1] - 5);
  ctx.restore();
}

function formatDetectionLabel(detection: VisionDetection, units: 'px' | 'mm' = 'px'): string {
  const confidence = `${(detection.confidence * 100).toFixed(0)}%`;
  const unitLabel = units === 'mm' ? 'mm' : 'px';
  if (detection.offset_xy != null && detection.distance != null) {
    const [dx, dy] = detection.offset_xy;
    return `${detection.class_name} ${confidence} | x:${dx.toFixed(1)} y:${dy.toFixed(1)} d:${detection.distance.toFixed(1)}${unitLabel}`;
  }
  if (detection.board_xy) {
    return `${detection.class_name} ${confidence} (${detection.board_xy[0].toFixed(2)}, ${detection.board_xy[1].toFixed(2)})`;
  }
  return `${detection.class_name} ${confidence}`;
}

export function drawDetectionOverlay(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  detections: VisionDetection[],
  fit: 'contain' | 'cover',
  sourceWidth: number,
  sourceHeight: number,
  inferenceFps?: number,
  error?: string | null,
  modelName?: string | null,
  workspace?: WorkspaceCalibration | null,
  pendingCalibrationPoints?: [number, number][],
  pendingCalibrationLabels?: readonly string[],
) {
  const containerWidth = canvas.clientWidth;
  const containerHeight = canvas.clientHeight;
  if (containerWidth <= 0 || containerHeight <= 0) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(containerWidth * dpr);
  canvas.height = Math.round(containerHeight * dpr);

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, containerWidth, containerHeight);

  const imageWidth = sourceWidth > 0 ? sourceWidth : image.naturalWidth;
  const imageHeight = sourceHeight > 0 ? sourceHeight : image.naturalHeight;
  const layout = getImageLayoutRect(containerWidth, containerHeight, imageWidth, imageHeight, fit);

  if (workspace) {
    drawWorkspace(ctx, workspace, layout);
  }

  if (pendingCalibrationPoints && pendingCalibrationPoints.length > 0) {
    pendingCalibrationPoints.forEach(([x, y], index) => {
      const [px, py] = mapPoint(x, y, layout);
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#a855f7';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
      const pointLabel = pendingCalibrationLabels?.[index] ?? `${index + 1}`;
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = '#e9d5ff';
      ctx.fillText(pointLabel, px + 10, py - 6);
    });
  }

  const origin = workspace?.origin_xy ?? workspace?.center_xy ?? null;
  const classColors: Record<string, string> = {
    'block': '#22c55e',
    'box': '#22c55e',
    'cube': '#22c55e',
    'mug': '#38bdf8',
    'cup': '#38bdf8',
  };
  const fallbackColors = ['#38bdf8', '#f97316', '#eab308', '#a855f7'];
  detections.forEach((detection, index) => {
    const color = classColors[detection.class_name] ?? fallbackColors[index % fallbackColors.length];
    drawObb(ctx, detection, layout, color);

    if (origin && detection.offset_xy != null) {
      const [originX, originY] = mapPoint(origin[0], origin[1], layout);
      const [targetX, targetY] = mapPoint(detection.center_xy[0], detection.center_xy[1], layout);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(originX, originY);
      ctx.lineTo(targetX, targetY);
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    const [labelX, labelY] = mapPoint(
      detection.bbox_xyxy[0],
      Math.max(0, detection.bbox_xyxy[1] - 4),
      layout,
    );
    const label = formatDetectionLabel(detection, workspace?.units ?? 'px');
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
    ctx.fillRect(labelX, labelY - 16, textWidth + 8, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, labelX + 4, labelY - 3);
  });

  ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
  ctx.fillRect(8, containerHeight - 28, 360, 20);
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  const backendLabel = modelName && modelName !== 'local-contrast' ? modelName : 'Local vision';
  const originLabel =
    workspace?.calibration_source === 'manual'
      ? 'manual | '
      : workspace?.calibration_source === 'apriltag'
        ? 'apriltag | '
        : workspace?.calibration_source === 'blue_dots'
          ? 'blue dots | '
          : workspace?.calibration_source === 'markers'
            ? '4 markers | '
            : workspace?.calibration_source === 'gripper'
              ? 'blue dot | '
              : workspace?.calibration_source === 'camera'
                ? 'camera cal | '
                : '';
  const calSource =
    workspace?.calibration_source === 'manual' ||
    workspace?.calibration_source === 'apriltag' ||
    workspace?.calibration_source === 'blue_dots' ||
    workspace?.calibration_source === 'markers' ||
    workspace?.calibration_source === 'camera'
      ? `${workspace.units ?? 'mm'} | `
      : workspace?.calibration_source === 'gripper'
        ? 'gripper px | '
        : workspace
          ? 'board px | '
          : '';
  const status = error
    ? `Vision error: ${error}`
    : `${calSource}${originLabel}${backendLabel}: ${detections.length} det${inferenceFps != null ? ` | ${inferenceFps.toFixed(1)} fps` : ''}`;
  ctx.fillText(status, 12, containerHeight - 14);
}
