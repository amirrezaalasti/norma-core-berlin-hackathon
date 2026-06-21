import type { GripperTipPosition, VisionDetection, WorkspaceCalibration } from './vision-types';
import { buildWorkspaceGridCells, buildWorkspaceGridLines } from './workspace-grid';

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

function drawWorkspaceGrid(
  ctx: CanvasRenderingContext2D,
  workspace: WorkspaceCalibration,
  layout: ImageLayoutRect,
) {
  const lines = buildWorkspaceGridLines(workspace);
  const cells = buildWorkspaceGridCells(workspace);

  ctx.save();
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
  ctx.lineWidth = 1;
  for (const line of lines) {
    const [x0, y0] = mapPoint(line.from[0], line.from[1], layout);
    const [x1, y1] = mapPoint(line.to[0], line.to[1], layout);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  for (const cell of cells) {
    const [cx, cy] = mapPoint(cell.center_pixel_xy[0], cell.center_pixel_xy[1], layout);
    const label = String(cell.square_id);
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
    ctx.fillRect(cx - textWidth / 2 - 4, cy - 8, textWidth + 8, 16);
    ctx.fillStyle = 'rgba(226, 232, 240, 0.95)';
    ctx.fillText(label, cx - textWidth / 2, cy + 4);
  }
  ctx.restore();
}

function drawDetectedGripperTip(
  ctx: CanvasRenderingContext2D,
  gripperTip: GripperTipPosition,
  layout: ImageLayoutRect,
  units: 'px' | 'mm' = 'mm',
) {
  const [px, py] = mapPoint(gripperTip.pixel_xy[0], gripperTip.pixel_xy[1], layout);
  ctx.save();
  ctx.beginPath();
  ctx.arc(px, py, 9, 0, Math.PI * 2);
  ctx.fillStyle = gripperTip.source === 'roboflow' || gripperTip.source === 'detected' ? '#eab308' : '#2563eb';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();

  const unitLabel = units === 'mm' ? 'mm' : 'px';
  const [dx, dy] = gripperTip.offset_xy;
  const label =
    gripperTip.source === 'roboflow' || gripperTip.source === 'detected'
      ? `yellow tape ${((gripperTip.confidence ?? 1) * 100).toFixed(0)}% | x:${dx.toFixed(1)} y:${dy.toFixed(1)}${unitLabel}`
      : `gripper tip | x:${dx.toFixed(1)} y:${dy.toFixed(1)}${unitLabel}`;
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  const textWidth = ctx.measureText(label).width;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
  ctx.fillRect(px + 10, py - 10, textWidth + 8, 16);
  ctx.fillStyle = gripperTip.source === 'roboflow' || gripperTip.source === 'detected' ? '#fde047' : '#60a5fa';
  ctx.fillText(label, px + 14, py + 2);
  ctx.restore();
}

function drawWorkspace(
  ctx: CanvasRenderingContext2D,
  workspace: WorkspaceCalibration,
  layout: ImageLayoutRect,
) {
  const corners = workspace.corners_xy.map(([x, y]) => mapPoint(x, y, layout));
  const isApriltag = workspace.calibration_source === 'apriltag';
  const isMarkers = workspace.calibration_source === 'markers';
  const isBlueDots = workspace.calibration_source === 'blue_dots';
  const isManual = workspace.calibration_source === 'manual';
  const isCamera = workspace.calibration_source === 'camera';
  const boardCenter = workspace.center_xy;
  const originPoint = isManual ? boardCenter : (workspace.origin_xy ?? boardCenter);
  const [originX, originY] = mapPoint(originPoint[0], originPoint[1], layout);
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

  if (isManual || isApriltag || isBlueDots || isMarkers || isCamera) {
    drawWorkspaceGrid(ctx, workspace, layout);
  }

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

  ctx.beginPath();
  ctx.arc(originX, originY, 7, 0, Math.PI * 2);
  ctx.fillStyle = isManual ? '#2563eb' : isFiducial ? '#2563eb' : isGripper ? '#38bdf8' : '#3b82f6';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();

  if (isManual) {
    const unitLabel = workspace.units === 'mm' ? 'mm' : 'px';
    const centerLabel = `center (0,0)${unitLabel}`;
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const centerTextWidth = ctx.measureText(centerLabel).width;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
    ctx.fillRect(originX + 10, originY - 10, centerTextWidth + 8, 16);
    ctx.fillStyle = '#60a5fa';
    ctx.fillText(centerLabel, originX + 14, originY + 2);
  }

  const label = isManual
    ? `manual 4-point center origin ${(workspace.confidence * 100).toFixed(0)}%`
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
  if (detection.square_id != null && detection.square_local_xy != null) {
    const [lx, ly] = detection.square_local_xy;
    const squarePart = `sq:${detection.square_id} (${lx.toFixed(2)},${ly.toFixed(2)})`;
    if (detection.offset_xy != null && detection.distance != null) {
      const [dx, dy] = detection.offset_xy;
      return `${detection.class_name} ${confidence} | ${squarePart} | x:${dx.toFixed(1)} y:${dy.toFixed(1)} d:${detection.distance.toFixed(1)}${unitLabel}`;
    }
    if (detection.board_xy) {
      return `${detection.class_name} ${confidence} | ${squarePart} (${detection.board_xy[0].toFixed(2)}, ${detection.board_xy[1].toFixed(2)})`;
    }
    return `${detection.class_name} ${confidence} | ${squarePart}`;
  }
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
  gripperTip?: GripperTipPosition | null,
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

  if (gripperTip?.pixel_xy) {
    drawDetectedGripperTip(ctx, gripperTip, layout, workspace?.units ?? 'mm');
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

  const origin =
    workspace?.calibration_source === 'manual'
      ? workspace.center_xy
      : workspace?.origin_xy ?? workspace?.center_xy ?? null;
  const classColors: Record<string, string> = {
    'block': '#22c55e',
    'box': '#22c55e',
    'cube': '#22c55e',
    'yellow_tape': '#eab308',
    'gripper_tip': '#eab308',
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
  const backendLabel =
    modelName && modelName !== 'local-contrast' && modelName !== 'local-color'
      ? modelName
      : 'Local color vision';
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
