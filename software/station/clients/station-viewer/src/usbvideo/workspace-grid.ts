import type { WorkspaceCalibration } from './vision-types';
import { boardNormalizedToPixel, pixelToBoardNormalized } from './workspace-detection';

export const BOARD_GRID_COLS = Number(import.meta.env.VITE_BOARD_GRID_COLS ?? 5);
export const BOARD_GRID_ROWS = Number(import.meta.env.VITE_BOARD_GRID_ROWS ?? 3);

export interface WorkspaceSquareInfo {
  /** 1-indexed square id (row-major from top-left of board). */
  square_id: number;
  square_col: number;
  square_row: number;
  /** Normalized board position of this square's center (0–1). */
  square_center_board_xy: [number, number];
  /** Offset from square center in cell units (−0.5…0.5 at cell edges). */
  square_local_xy: [number, number];
}

export interface WorkspaceGridLine {
  /** Pixel endpoints for overlay drawing. */
  from: [number, number];
  to: [number, number];
}

export interface WorkspaceGridCell {
  square_id: number;
  square_col: number;
  square_row: number;
  center_board_xy: [number, number];
  center_pixel_xy: [number, number];
  /** Four corners in pixel space (TL, TR, BR, BL within the cell). */
  corners_pixel_xy: [[number, number], [number, number], [number, number], [number, number]];
}

function clampGridDimension(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value < 1) {
    return fallback;
  }
  return Math.floor(value);
}

export function workspaceGridDimensions(): { cols: number; rows: number } {
  return {
    cols: clampGridDimension(BOARD_GRID_COLS, 5),
    rows: clampGridDimension(BOARD_GRID_ROWS, 3),
  };
}

/** Map normalized board coordinates to a grid square and local offset from its center. */
export function squareInfoFromBoardXy(
  boardXy: [number, number],
  cols = workspaceGridDimensions().cols,
  rows = workspaceGridDimensions().rows,
): WorkspaceSquareInfo {
  const [u, v] = boardXy;
  const col = Math.min(cols - 1, Math.max(0, Math.floor(u * cols)));
  const row = Math.min(rows - 1, Math.max(0, Math.floor(v * rows)));
  const centerU = (col + 0.5) / cols;
  const centerV = (row + 0.5) / rows;
  return {
    square_id: row * cols + col + 1,
    square_col: col,
    square_row: row,
    square_center_board_xy: [centerU, centerV],
    square_local_xy: [(u - centerU) * cols, (v - centerV) * rows],
  };
}

export function enrichDetectionWithSquare(
  boardXy: [number, number] | null | undefined,
): WorkspaceSquareInfo | null {
  if (!boardXy) {
    return null;
  }
  return squareInfoFromBoardXy(boardXy);
}

/** Vertical and horizontal grid lines in pixel space for overlay drawing. */
export function buildWorkspaceGridLines(
  workspace: WorkspaceCalibration,
  cols = workspaceGridDimensions().cols,
  rows = workspaceGridDimensions().rows,
): WorkspaceGridLine[] {
  const lines: WorkspaceGridLine[] = [];

  for (let i = 1; i < cols; i += 1) {
    const u = i / cols;
    const from = boardNormalizedToPixel(u, 0, workspace);
    const to = boardNormalizedToPixel(u, 1, workspace);
    if (from && to) {
      lines.push({ from, to });
    }
  }

  for (let j = 1; j < rows; j += 1) {
    const v = j / rows;
    const from = boardNormalizedToPixel(0, v, workspace);
    const to = boardNormalizedToPixel(1, v, workspace);
    if (from && to) {
      lines.push({ from, to });
    }
  }

  return lines;
}

export function buildWorkspaceGridCells(
  workspace: WorkspaceCalibration,
  cols = workspaceGridDimensions().cols,
  rows = workspaceGridDimensions().rows,
): WorkspaceGridCell[] {
  const cells: WorkspaceGridCell[] = [];

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const u0 = col / cols;
      const u1 = (col + 1) / cols;
      const v0 = row / rows;
      const v1 = (row + 1) / rows;
      const centerU = (col + 0.5) / cols;
      const centerV = (row + 0.5) / rows;

      const tl = boardNormalizedToPixel(u0, v0, workspace);
      const tr = boardNormalizedToPixel(u1, v0, workspace);
      const br = boardNormalizedToPixel(u1, v1, workspace);
      const bl = boardNormalizedToPixel(u0, v1, workspace);
      const center = boardNormalizedToPixel(centerU, centerV, workspace);
      if (!tl || !tr || !br || !bl || !center) {
        continue;
      }

      cells.push({
        square_id: row * cols + col + 1,
        square_col: col,
        square_row: row,
        center_board_xy: [centerU, centerV],
        center_pixel_xy: center,
        corners_pixel_xy: [tl, tr, br, bl],
      });
    }
  }

  return cells;
}

/** Board center (average of four manual corners) in normalized coordinates — should be ~(0.5, 0.5). */
export function boardCenterBoardXy(workspace: WorkspaceCalibration): [number, number] | null {
  return pixelToBoardNormalized(workspace.center_xy[0], workspace.center_xy[1], workspace);
}
