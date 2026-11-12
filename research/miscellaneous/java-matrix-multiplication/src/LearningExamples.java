import java.util.Arrays;

/** Executable double-array examples shown on the page. No external libraries. */
public final class LearningExamples {
    private LearningExamples() {}

    public static double[][] multiply(double[][] a, double[][] b) {
        validate(a); validate(b);
        if (a[0].length != b.length) throw new IllegalArgumentException("inner dimensions differ");
        double[][] c = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < b.length; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }

    public static double sum(double[] values, boolean reverse) {
        double result = 0;
        for (int k = 0; k < values.length; k++) result += values[reverse ? values.length - 1 - k : k];
        return result;
    }

    private static void validate(double[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
            throw new IllegalArgumentException("empty matrix");
        for (double[] row : matrix) if (row == null || row.length != matrix[0].length)
            throw new IllegalArgumentException("ragged matrix");
    }

    public static void main(String[] args) {
        double[][] A = {{2, -1, 3}, {4, 0, 1}};
        double[][] B = {{1, 5}, {2, -2}, {0, 3}};
        double[][] C = multiply(A, B);
        System.out.println(Arrays.deepToString(C));
        double[] terms = {0.1, 0.2, 0.3};
        System.out.println("Forward: " + sum(terms, false));
        System.out.println("Reverse: " + sum(terms, true));
    }
}
