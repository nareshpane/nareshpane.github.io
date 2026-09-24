import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

/** Observes real kernels and serializes deterministic teaching data, never timings. */
public final class TraceWriter {
    static final int[][] A = {{2,-1,3},{4,0,1}}, B = {{1,5},{2,-2},{0,3}};
    static final int[][] TILE_A = {{1,2,0,1},{0,1,3,2},{2,0,1,1},{1,1,0,2}};
    static final int[][] TILE_B = {{2,0,1,1},{1,2,0,1},{0,1,2,0},{1,0,1,2}};

    public static void main(String[] args) throws IOException {
        Path output = args.length == 0 ? Paths.get("traces") : Paths.get(args[0]);
        Files.createDirectories(output);
        write(output, "row-column", rowColumnTrace());
        write(output, "blocked", blockedTrace());
        write(output, "strassen", strassenTrace());
        write(output, "parallel", parallelTrace());
        write(output, "learning", learningTrace());
        System.out.println("Wrote 5 deterministic Java traces to " + output.toAbsolutePath());
    }

    static String rowColumnTrace() {
        List<String> events = new ArrayList<>();
        int[][] c = MatrixAlgorithms.classical(A, B, (i,j,k,product,partial) -> {
            events.add("{\"i\":"+i+",\"j\":"+j+",\"k\":"+k+",\"left\":"+A[i][k]
                +",\"right\":"+B[k][j]+",\"product\":"+product+",\"partial\":"+partial
                +",\"cellComplete\":"+(k==B.length-1)+"}");
        });
        return base("classical-row-column", A, B, c)+",\"events\":["+String.join(",\n",events)+"]}";
    }

    static String blockedTrace() {
        List<String> events = new ArrayList<>();
        int[][] c = MatrixAlgorithms.blocked(TILE_A, TILE_B, 2, (ii,kk,jj,partial) -> {
            events.add("{\"ii\":"+ii+",\"kk\":"+kk+",\"jj\":"+jj
                +",\"outputTile\":["+ii/2+","+jj/2+"],\"aTile\":["+ii/2+","+kk/2
                +"],\"bTile\":["+kk/2+","+jj/2+"],\"partial\":"+matrix(partial)+"}");
        });
        return base("blocked-ii-kk-jj",TILE_A,TILE_B,c)+",\"tileSize\":2,\"events\":["+String.join(",\n",events)+"]}";
    }

    static String strassenTrace() {
        int[][] a={{3,1},{2,4}}, b={{2,0},{1,5}};
        int[][] c=MatrixAlgorithms.strassen(a,b);
        int[] values=MatrixAlgorithms.strassenProducts(a,b);
        String[] formulas={"(a+d)(e+h)","(c+d)e","a(f-h)","d(g-e)","(a+b)h","(c-a)(e+f)","(b-d)(g+h)"};
        List<String> products=new ArrayList<>();
        for(int i=0;i<7;i++) products.add("{\"name\":\"M"+(i+1)+"\",\"formula\":\""+formulas[i]+"\",\"value\":"+values[i]+"}");
        return base("strassen-2x2",a,b,c)+",\"verified\":"+MatrixAlgorithms.equals(c,MatrixAlgorithms.classical(a,b))
            +",\"products\":["+String.join(",",products)+"]}";
    }

    static String parallelTrace() {
        int[][] c=MatrixAlgorithms.parallel(TILE_A,TILE_B,2,4);
        List<String> tasks=new ArrayList<>();
        // Lane assignment illustrates tasks; actual worker ownership is nondeterministic.
        for(int lane=0;lane<4;lane++) {
            int r=lane/2*2, col=lane%2*2;
            int[][] tile={{c[r][col],c[r][col+1]},{c[r+1][col],c[r+1][col+1]}};
            tasks.add("{\"lane\":"+lane+",\"row\":"+r+",\"col\":"+col+",\"result\":"+matrix(tile)+"}");
        }
        return base("ForkJoinPool-output-tiles",TILE_A,TILE_B,c)+",\"verified\":"+MatrixAlgorithms.equals(c,MatrixAlgorithms.classical(TILE_A,TILE_B))
            +",\"note\":\"Illustrative lane assignment; values computed by actual ForkJoinPool tasks.\",\"tasks\":["+String.join(",",tasks)+"]}";
    }

    static String learningTrace() {
        int[][] shear={{1,1},{0,1}}, rotate={{0,-1},{1,0}}, x={{1},{1}};
        int[][] ab=MatrixAlgorithms.classical(shear,rotate), ba=MatrixAlgorithms.classical(rotate,shear);
        int[][] sparse={{0,4,0,0,0,0},{0,0,0,-2,0,0},{3,0,0,0,0,0},{0,0,5,0,0,1},{0,0,0,0,7,0},{0,2,0,0,0,0}};
        MatrixAlgorithms.CsrMatrix csr=MatrixAlgorithms.CsrMatrix.fromDense(sparse);
        double[] terms={0.1,0.2,0.3};
        List<String> scale=new ArrayList<>();
        for(int n=2;n<=64;n*=2) scale.add("{\"n\":"+n+",\"cells\":"+(n*n)+",\"products\":"+(n*n*n)+"}");
        return "{\"schema\":2,\"geometry\":{\"a\":"+matrix(shear)+",\"b\":"+matrix(rotate)+",\"ab\":"+matrix(ab)+",\"ba\":"+matrix(ba)
            +",\"x\":"+matrix(x)+",\"bx\":"+matrix(MatrixAlgorithms.classical(rotate,x))+",\"ax\":"+matrix(MatrixAlgorithms.classical(shear,x))
            +",\"abx\":"+matrix(MatrixAlgorithms.classical(ab,x))+",\"bax\":"+matrix(MatrixAlgorithms.classical(ba,x))+"},\"scale\":["+String.join(",",scale)
            +"],\"sparse\":{\"matrix\":"+matrix(sparse)+",\"values\":"+Arrays.toString(csr.values)+",\"columns\":"+Arrays.toString(csr.columnIndices)+",\"rowOffsets\":"+Arrays.toString(csr.rowOffsets)
            +"},\"precision\":{\"terms\":"+Arrays.toString(terms)+",\"forward\":\""+LearningExamples.sum(terms,false)+"\",\"reverse\":\""+LearningExamples.sum(terms,true)
            +"\",\"cancelFirst\":"+((1e16+-1e16)+1)+",\"smallFirst\":"+(1e16+(-1e16+1))+"}}";
    }

    static String base(String name,int[][] a,int[][] b,int[][] c) {
        return "{\"schema\":2,\"algorithm\":\""+name+"\",\"a\":"+matrix(a)+",\"b\":"+matrix(b)+",\"result\":"+matrix(c);
    }
    static String matrix(int[][] m) { return Arrays.deepToString(m); }
    private static void write(Path output,String name,String data) throws IOException {
        Files.write(output.resolve(name+".json"),(data+"\n").getBytes(StandardCharsets.UTF_8));
    }
}
