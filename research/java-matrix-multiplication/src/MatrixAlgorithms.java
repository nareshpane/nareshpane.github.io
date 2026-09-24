import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/** Integer matrix algorithms used by the page and its deterministic tests. */
public final class MatrixAlgorithms {
    private MatrixAlgorithms() {}

    @FunctionalInterface
    public interface ScalarObserver { void accept(int i, int j, int k, int product, int partial); }
    @FunctionalInterface
    public interface TileObserver { void accept(int ii, int kk, int jj, int[][] partial); }

    public static int[][] classical(int[][] a, int[][] b) {
        return classical(a, b, null);
    }

    public static int[][] classical(int[][] a, int[][] b, ScalarObserver observer) {
        requireMultipliable(a, b);
        int[][] c = new int[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < b.length; k++) {
                    int product = a[i][k] * b[k][j];
                    c[i][j] += product;
                    if (observer != null) observer.accept(i, j, k, product, c[i][j]);
                }
            }
        }
        return c;
    }

    public static int[][] blocked(int[][] a, int[][] b, int blockSize) {
        return blocked(a, b, blockSize, null);
    }

    public static int[][] blocked(int[][] a, int[][] b, int blockSize, TileObserver observer) {
        requireMultipliable(a, b);
        if (blockSize < 1) throw new IllegalArgumentException("blockSize must be positive");
        int rows = a.length, shared = b.length, cols = b[0].length;
        int[][] c = new int[rows][cols];
        for (int ii = 0; ii < rows; ii += blockSize) {
            for (int kk = 0; kk < shared; kk += blockSize) {
                for (int jj = 0; jj < cols; jj += blockSize) {
                    for (int i = ii; i < Math.min(ii + blockSize, rows); i++) {
                        for (int k = kk; k < Math.min(kk + blockSize, shared); k++) {
                            int aik = a[i][k];
                            for (int j = jj; j < Math.min(jj + blockSize, cols); j++) {
                                c[i][j] += aik * b[k][j];
                            }
                        }
                    }
                    if (observer != null) observer.accept(ii, kk, jj, c);
                }
            }
        }
        return c;
    }

    /** Strassen multiplication for equal square matrices whose size is a power of two. */
    public static int[][] strassen(int[][] a, int[][] b) {
        requireMultipliable(a, b);
        int n = a.length;
        if (n != a[0].length || n != b.length || n != b[0].length || (n & (n - 1)) != 0) {
            throw new IllegalArgumentException("Strassen input must be equal square power-of-two matrices");
        }
        return strassenRecursive(a, b);
    }

    private static int[][] strassenRecursive(int[][] a, int[][] b) {
        int n = a.length;
        if (n <= 2) return strassenBase(a, b);
        int h = n / 2;
        int[][] a11 = slice(a, 0, 0, h), a12 = slice(a, 0, h, h);
        int[][] a21 = slice(a, h, 0, h), a22 = slice(a, h, h, h);
        int[][] b11 = slice(b, 0, 0, h), b12 = slice(b, 0, h, h);
        int[][] b21 = slice(b, h, 0, h), b22 = slice(b, h, h, h);
        int[][] m1 = strassenRecursive(add(a11, a22), add(b11, b22));
        int[][] m2 = strassenRecursive(add(a21, a22), b11);
        int[][] m3 = strassenRecursive(a11, subtract(b12, b22));
        int[][] m4 = strassenRecursive(a22, subtract(b21, b11));
        int[][] m5 = strassenRecursive(add(a11, a12), b22);
        int[][] m6 = strassenRecursive(subtract(a21, a11), add(b11, b12));
        int[][] m7 = strassenRecursive(subtract(a12, a22), add(b21, b22));
        int[][] c11 = add(subtract(add(m1, m4), m5), m7);
        int[][] c12 = add(m3, m5);
        int[][] c21 = add(m2, m4);
        int[][] c22 = add(subtract(add(m1, m3), m2), m6);
        return join(c11, c12, c21, c22);
    }

    private static int[][] strassenBase(int[][] a, int[][] b) {
        if (a.length == 1) return new int[][] {{a[0][0] * b[0][0]}};
        int[] m = strassenProducts(a, b);
        return new int[][] {
            {m[0] + m[3] - m[4] + m[6], m[2] + m[4]},
            {m[1] + m[3], m[0] - m[1] + m[2] + m[5]}
        };
    }

    public static int[] strassenProducts(int[][] a, int[][] b) {
        int x = a[0][0], y = a[0][1], z = a[1][0], w = a[1][1];
        int e = b[0][0], f = b[0][1], g = b[1][0], h = b[1][1];
        int m1 = (x + w) * (e + h), m2 = (z + w) * e;
        int m3 = x * (f - h), m4 = w * (g - e);
        int m5 = (x + y) * h, m6 = (z - x) * (e + f), m7 = (y - w) * (g + h);
        return new int[] {m1, m2, m3, m4, m5, m6, m7};
    }

    public static int[][] parallel(int[][] a, int[][] b, int tileSize, int parallelism) {
        requireMultipliable(a, b);
        if (tileSize < 1 || parallelism < 1) throw new IllegalArgumentException("positive tile size and parallelism required");
        int[][] c = new int[a.length][b[0].length];
        List<RecursiveAction> tasks = new ArrayList<RecursiveAction>();
        for (int row = 0; row < a.length; row += tileSize) {
            for (int col = 0; col < b[0].length; col += tileSize) {
                final int r0 = row, c0 = col;
                tasks.add(new RecursiveAction() {
                    protected void compute() {
                        for (int i = r0; i < Math.min(r0 + tileSize, a.length); i++) {
                            for (int j = c0; j < Math.min(c0 + tileSize, b[0].length); j++) {
                                int sum = 0;
                                for (int k = 0; k < b.length; k++) sum += a[i][k] * b[k][j];
                                c[i][j] = sum;
                            }
                        }
                    }
                });
            }
        }
        ForkJoinPool pool = new ForkJoinPool(parallelism);
        try {
            pool.submit(new RecursiveAction() {
                protected void compute() { invokeAll(tasks); }
            }).join();
        } finally {
            pool.shutdown();
        }
        return c;
    }

    /** Minimal compressed-row sparse matrix used for the worked sparse example. */
    public static final class CsrMatrix {
        public final int rows, cols;
        public final int[] rowOffsets, columnIndices, values;

        private CsrMatrix(int rows, int cols, int[] rowOffsets, int[] columnIndices, int[] values) {
            this.rows = rows; this.cols = cols; this.rowOffsets = rowOffsets;
            this.columnIndices = columnIndices; this.values = values;
        }

        public static CsrMatrix fromDense(int[][] dense) {
            validate(dense);
            int count = 0;
            for (int[] row : dense) for (int value : row) if (value != 0) count++;
            int[] offsets = new int[dense.length + 1], cols = new int[count], vals = new int[count];
            int p = 0;
            for (int i = 0; i < dense.length; i++) {
                offsets[i] = p;
                for (int j = 0; j < dense[0].length; j++) if (dense[i][j] != 0) {
                    cols[p] = j; vals[p] = dense[i][j]; p++;
                }
            }
            offsets[dense.length] = p;
            return new CsrMatrix(dense.length, dense[0].length, offsets, cols, vals);
        }
    }

    /** Sparse-left by dense-right multiplication; zero entries are never visited. */
    public static int[][] sparseTimesDense(CsrMatrix a, int[][] b) {
        validate(b);
        if (a.cols != b.length) throw new IllegalArgumentException("inner dimensions differ");
        int[][] c = new int[a.rows][b[0].length];
        for (int i = 0; i < a.rows; i++) {
            for (int p = a.rowOffsets[i]; p < a.rowOffsets[i + 1]; p++) {
                int k = a.columnIndices[p], value = a.values[p];
                for (int j = 0; j < b[0].length; j++) c[i][j] += value * b[k][j];
            }
        }
        return c;
    }

    public static boolean equals(int[][] a, int[][] b) {
        if (a.length != b.length || a[0].length != b[0].length) return false;
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++)
            if (a[i][j] != b[i][j]) return false;
        return true;
    }

    private static int[][] add(int[][] a, int[][] b) {
        int[][] out = new int[a.length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a.length; j++) out[i][j] = a[i][j] + b[i][j];
        return out;
    }

    private static int[][] subtract(int[][] a, int[][] b) {
        int[][] out = new int[a.length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a.length; j++) out[i][j] = a[i][j] - b[i][j];
        return out;
    }

    private static int[][] slice(int[][] a, int row, int col, int size) {
        int[][] out = new int[size][size];
        for (int i = 0; i < size; i++) System.arraycopy(a[row + i], col, out[i], 0, size);
        return out;
    }

    private static int[][] join(int[][] a11, int[][] a12, int[][] a21, int[][] a22) {
        int h = a11.length, n = h * 2;
        int[][] out = new int[n][n];
        for (int i = 0; i < h; i++) {
            System.arraycopy(a11[i], 0, out[i], 0, h); System.arraycopy(a12[i], 0, out[i], h, h);
            System.arraycopy(a21[i], 0, out[h + i], 0, h); System.arraycopy(a22[i], 0, out[h + i], h, h);
        }
        return out;
    }

    private static void requireMultipliable(int[][] a, int[][] b) {
        validate(a); validate(b);
        if (a[0].length != b.length) throw new IllegalArgumentException("inner dimensions differ");
    }

    private static void validate(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0) throw new IllegalArgumentException("empty matrix");
        int width = matrix[0].length;
        for (int[] row : matrix) if (row == null || row.length != width) throw new IllegalArgumentException("ragged matrix");
    }
}
