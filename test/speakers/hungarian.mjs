// Maximum-sum assignment on a rectangular matrix (like scipy.optimize.linear_sum_assignment(maximize=True)):
// the Hungarian algorithm with potentials, O(n^2 m). Returns [row, col] pairs, min(rows, cols) of them.
export function maxAssignment(matrix) {
  const rows = matrix.length;
  const cols = rows ? matrix[0].length : 0;
  if (!rows || !cols) return [];
  const transposed = rows > cols;
  const n = transposed ? cols : rows; // n <= m
  const m = transposed ? rows : cols;
  const cost = (i, j) => -(transposed ? matrix[j][i] : matrix[i][j]);
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1); // p[j]: the row (1-based) assigned to column j
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const out = [];
  for (let j = 1; j <= m; j++) if (p[j]) out.push(transposed ? [j - 1, p[j] - 1] : [p[j] - 1, j - 1]);
  return out.sort((a, b) => a[0] - b[0]);
}
