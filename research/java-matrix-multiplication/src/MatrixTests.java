import java.util.Random;

/** Dependency-free correctness checks. Exits by throwing AssertionError on failure. */
public final class MatrixTests {
    public static void main(String[] args) {
        int[][] a = {{2, -1, 3}, {4, 0, 1}};
        int[][] b = {{1, 5}, {2, -2}, {0, 3}};
        assertMatrix("worked example", new int[][] {{0, 21}, {4, 23}}, MatrixAlgorithms.classical(a, b));

        Random random = new Random(20260922L);
        int[][] x = randomMatrix(random, 8, 8), y = randomMatrix(random, 8, 8);
        int[][] expected = MatrixAlgorithms.classical(x, y);
        assertMatrix("blocked == classical", expected, MatrixAlgorithms.blocked(x, y, 3));
        assertMatrix("Strassen == classical", expected, MatrixAlgorithms.strassen(x, y));
        assertMatrix("parallel == classical", expected, MatrixAlgorithms.parallel(x, y, 2, 4));

        int[][] sparse = {{0, 0, 3, 0}, {2, 0, 0, 0}, {0, -1, 0, 4}, {0, 0, 0, 0}};
        int[][] dense = {{1, 2}, {3, 1}, {2, 5}, {-1, 2}};
        assertMatrix("sparse == classical", MatrixAlgorithms.classical(sparse, dense),
            MatrixAlgorithms.sparseTimesDense(MatrixAlgorithms.CsrMatrix.fromDense(sparse), dense));
        for (int size : new int[] {1,2,4,16}) {
            int[][] left=randomMatrix(random,size,size), right=randomMatrix(random,size,size);
            assertMatrix("Strassen size " + size, MatrixAlgorithms.classical(left,right), MatrixAlgorithms.strassen(left,right));
        }
        int[][] rectangular=randomMatrix(random,3,5), right=randomMatrix(random,5,7);
        for (int tile : new int[] {1,2,4,9}) {
            assertMatrix("rectangular blocked tile " + tile,MatrixAlgorithms.classical(rectangular,right),MatrixAlgorithms.blocked(rectangular,right,tile));
            assertMatrix("rectangular parallel tile " + tile,MatrixAlgorithms.classical(rectangular,right),MatrixAlgorithms.parallel(rectangular,right,tile,3));
        }
        int[] seen={0};
        MatrixAlgorithms.classical(a,b,(i,j,k,product,partial)->seen[0]++);
        if (seen[0]!=12) throw new AssertionError("observer missed arithmetic");
        int[] tileStep={0};
        MatrixAlgorithms.blocked(TraceWriter.TILE_A,TraceWriter.TILE_B,2,(ii,kk,jj,partial)->{
            int step=tileStep[0]++;
            if (ii!=step/4*2 || kk!=(step/2)%2*2 || jj!=step%2*2) throw new AssertionError("tile trace order");
        });
        if(tileStep[0]!=8) throw new AssertionError("tile event count");
        double[][] doubles=LearningExamples.multiply(new double[][]{{2,-1,3},{4,0,1}},new double[][]{{1,5},{2,-2},{0,3}});
        if(doubles[0][1]!=21 || doubles[1][1]!=23) throw new AssertionError("double example");
        double[] terms={.1,.2,.3};
        if(LearningExamples.sum(terms,false)==LearningExamples.sum(terms,true)) throw new AssertionError("precision example");
        rejects(()->MatrixAlgorithms.classical(new int[][]{null},b));
        rejects(()->MatrixAlgorithms.classical(new int[][]{{1},{1,2}},b));
        rejects(()->MatrixAlgorithms.classical(new int[][]{{1}},b));
        rejects(()->MatrixAlgorithms.blocked(a,b,0));
        rejects(()->MatrixAlgorithms.parallel(a,b,2,0));
        rejects(()->MatrixAlgorithms.strassen(a,b));
        System.out.println("PASS: trace order, double arithmetic, invalid inputs, and all algorithms.");
    }

    private static void rejects(Runnable operation) {
        try { operation.run(); } catch (IllegalArgumentException expected) { return; }
        throw new AssertionError("invalid input accepted");
    }

    private static int[][] randomMatrix(Random random, int rows, int cols) {
        int[][] matrix = new int[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) matrix[i][j] = random.nextInt(11) - 5;
        return matrix;
    }

    private static void assertMatrix(String name, int[][] expected, int[][] actual) {
        if (!MatrixAlgorithms.equals(expected, actual)) throw new AssertionError(name + " failed");
        System.out.println("PASS: " + name);
    }
}
