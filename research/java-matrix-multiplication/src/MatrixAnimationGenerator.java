import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.GradientPaint;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;

/** Renders the poster, recap image, and every frame of the documentary MP4 with Java2D. */
public final class MatrixAnimationGenerator {
    private static final int W = 1280, H = 720, FPS = 20, SECONDS = 30, FRAMES = FPS * SECONDS;
    private static final Color BG = new Color(247, 244, 238), PANEL = new Color(255, 249, 236);
    private static final Color INK = new Color(48, 45, 37), MUTED = new Color(108, 96, 77);
    private static final Color CYAN = new Color(53, 104, 93), GOLD = new Color(139, 100, 31);
    private static final Color CORAL = new Color(163, 73, 48), GRID = new Color(214, 198, 170);
    private static final Font TITLE = new Font("Serif", Font.BOLD, 46), HEAD = new Font("SansSerif", Font.BOLD, 25);
    private static final Font BODY = new Font("SansSerif", Font.PLAIN, 18), MONO = new Font("Monospaced", Font.BOLD, 22);

    public static void main(String[] args) throws IOException {
        Path frames = args.length > 0 ? Paths.get(args[0]) : Paths.get("build/frames");
        Path images = args.length > 1 ? Paths.get(args[1]) : Paths.get("images");
        Files.createDirectories(frames); Files.createDirectories(images);
        BufferedImage poster = render(68);
        ImageIO.write(poster, "png", images.resolve("matrix-motion-poster.png").toFile());
        ImageIO.write(recap(), "png", images.resolve("algorithm-recap.png").toFile());
        for (int frame = 0; frame < FRAMES; frame++) {
            ImageIO.write(render(frame), "png", frames.resolve(String.format("frame-%04d.png", frame)).toFile());
            if (frame % 100 == 0) System.out.println("Rendered frame " + frame + " / " + FRAMES);
        }
        System.out.println("Rendered " + FRAMES + " frames at " + W + "x" + H + " (" + FPS + " fps). ");
    }

    private static BufferedImage render(int frame) {
        BufferedImage image = canvas(); Graphics2D g = image.createGraphics(); setup(g);
        background(g); header(g, frame);
        int stage = Math.min(6, frame / 86), local = frame % 86;
        switch (stage) {
            case 0: rowColumn(g, local); break;
            case 1: manyCells(g, local); break;
            case 2: largerMatrix(g, local); break;
            case 3: tiled(g, local); break;
            case 4: strassen(g, local); break;
            case 5: parallel(g, local); break;
            default: finalMatrix(g, local); break;
        }
        timeline(g, frame); g.dispose(); return image;
    }

    private static BufferedImage recap() {
        BufferedImage image = canvas(); Graphics2D g = image.createGraphics(); setup(g); background(g);
        g.setFont(TITLE); text(g, "One operation, seven ways to see it", 64, 82, INK);
        String[] labels = {"row × column", "classical loops", "cache tiles", "7-way recursion", "skip zeros", "worker tasks", "exponent ω"};
        Color[] colors = {CYAN, GOLD, CYAN, CORAL, GOLD, CYAN, CORAL};
        int x = 54, y = 278;
        for (int i = 0; i < labels.length; i++) {
            int w = i == 0 ? 158 : 145;
            rounded(g, x, y, w, 112, PANEL, colors[i]);
            g.setFont(new Font("SansSerif", Font.BOLD, 17)); center(g, labels[i], x + w / 2, y + 48, INK);
            g.setFont(new Font("SansSerif", Font.PLAIN, 14)); center(g, String.format("%02d", i + 1), x + w / 2, y + 78, MUTED);
            if (i < labels.length - 1) { text(g, "→", x + w + 11, y + 66, GOLD); }
            x += w + 30;
        }
        g.setFont(BODY); text(g, "Arithmetic stays exact in this integer example; the organization of work changes.", 64, 474, MUTED);
        drawMatrix(g, new int[][]{{0,21},{4,23}}, 520, 520, 78, -1, -1, true);
        g.dispose(); return image;
    }

    private static void rowColumn(Graphics2D g, int t) {
        label(g, "01", "ROW MEETS COLUMN", "Each output cell is one dot product.");
        int[][] a={{2,-1,3},{4,0,1}}, b={{1,5},{2,-2},{0,3}};
        int k = Math.min(2, t / 24), pulse = t % 24;
        drawMatrix(g,a,120,250,74,0,k,false); symbol(g,"×",390,330); drawMatrix(g,b,475,210,74,k,0,false); symbol(g,"=",720,330);
        drawMatrix(g,new int[][]{{0,21},{4,23}},810,250,74,0,0,true);
        int ax=120+k*74+37, ay=250+37, bx=475+37, by=210+k*74+37;
        double p = Math.min(1, pulse/18.0); int px=(int)(ax+(660-ax)*p), py=(int)(ay+(330-ay)*p);
        int qx=(int)(bx+(660-bx)*p), qy=(int)(by+(330-by)*p);
        dot(g,px,py,CYAN); dot(g,qx,qy,GOLD); g.setFont(MONO); center(g,a[0][k]+" × "+b[k][0],660,410,INK);
    }

    private static void manyCells(Graphics2D g, int t) {
        label(g,"02","MANY DOT PRODUCTS","The same rule sweeps across every output position.");
        int active=(t/8)%16; int[][] m=new int[4][4]; for(int i=0;i<4;i++)for(int j=0;j<4;j++)m[i][j]=(i+1)*(j+2);
        drawMatrix(g,m,430,190,78,active/4,active%4,true);
        for(int q=0;q<9;q++){double a=(t*.09+q*.7);dot(g,(int)(640+Math.cos(a)*260),(int)(360+Math.sin(a*1.3)*190),q%2==0?CYAN:GOLD);}
        g.setFont(HEAD); center(g,"16 output cells • 64 scalar products",640,580,INK);
    }

    private static void largerMatrix(Graphics2D g,int t){
        label(g,"03","SCALE CHANGES THE VIEW","Individual cells become a field of coordinated work.");
        int n=10, size=38, x=450,y=150; int active=(t/3)%(n*n);
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){Color fill=(i*n+j<=active)?new Color(190+(i*4)%30,205+(j*3)%30,166):PANEL;cell(g,x+j*size,y+i*size,size,fill,GRID);}
        g.setFont(BODY); center(g,"A 10 × 10 result contains 100 dot products",640,590,MUTED);
    }

    private static void tiled(Graphics2D g,int t){
        label(g,"04","TILES REUSE NEARBY DATA","Blocking changes movement through memory, not cubic arithmetic count.");
        int size=52,x=180,y=180,n=6; for(int i=0;i<n;i++)for(int j=0;j<n;j++){Color f=((i/2+j/2)%2==0)?new Color(227,236,223):new Color(246,237,200);cell(g,x+j*size,y+i*size,size,f,GRID);}
        int tile=(t/14)%9, ti=tile/3,tj=tile%3; outline(g,x+tj*104,y+ti*104,104,104,GOLD,5);
        rounded(g,810,230,270,185,new Color(246,237,200),CYAN); g.setFont(HEAD); center(g,"CACHE",945,275,CYAN);
        int sx=x+tj*104+52,sy=y+ti*104+52; double p=Math.min(1,(t%14)/10.0); line(g,sx,sy,(int)(sx+(810-sx)*p),(int)(sy+(320-sy)*p),GOLD,5);
        g.setFont(BODY); center(g,"reuse a small working set",945,356,INK);
    }

    private static void strassen(Graphics2D g,int t){
        label(g,"05","SEVEN BRANCHES","Strassen trades one block multiplication for more additions and subtraction.");
        int cx=640,cy=210; rounded(g,cx-100,cy-38,200,76,PANEL,CORAL); g.setFont(HEAD); center(g,"2 × 2 blocks",cx,cy+7,INK);
        int shown=Math.min(7,t/10+1); for(int i=0;i<7;i++){double angle=Math.PI*.12+Math.PI*.76*i/6;int x=(int)(640+420*Math.cos(angle)),y=(int)(265+270*Math.sin(angle));line(g,cx,cy+38,x,y-28,i<shown?CORAL:GRID,i<shown?4:2);rounded(g,x-45,y-27,90,54,i<shown?new Color(250,230,214):PANEL,i<shown?CORAL:GRID);g.setFont(MONO);center(g,"M"+(i+1),x,y+7,i<shown?INK:MUTED);}
        g.setFont(BODY); center(g,"8 ordinary block products → 7 recursive products",640,625,MUTED);
    }

    private static void parallel(Graphics2D g,int t){
        label(g,"06","INDEPENDENT TILES BECOME TASKS","Worker lanes are a conceptual schedule, not a picture of CPU electronics.");
        for(int lane=0;lane<4;lane++){int y=190+lane*90;rounded(g,175,y,930,62,new Color(244,237,219),GRID);g.setFont(BODY);text(g,"Java worker "+(lane+1),195,y+38,INK);double p=((t+lane*9)%70)/70.0;int x=(int)(420+p*620);rounded(g,x,y+9,105,44,lane%2==0?new Color(227,236,223):new Color(246,237,200),lane%2==0?CYAN:GOLD);g.setFont(new Font("Monospaced",Font.BOLD,16));center(g,"tile "+(lane/2)+","+(lane%2),x+52,y+37,INK);}
    }

    private static void finalMatrix(Graphics2D g,int t){
        label(g,"07","RESULT","The worked example returns: [[0, 21], [4, 23]].");
        int[][] c=MatrixAlgorithms.classical(TraceWriter.A,TraceWriter.B);drawMatrix(g,c,490,245,150,-1,-1,true);
        double alpha=Math.min(1,t/50.0); g.setFont(TITLE); center(g,"C = A B",640,650,new Color(CYAN.getRed(),CYAN.getGreen(),CYAN.getBlue(),(int)(255*alpha)));
    }

    private static void header(Graphics2D g,int frame){g.setFont(new Font("SansSerif",Font.BOLD,14));text(g,"MATRIX MULTIPLICATION IN MOTION",54,42,MUTED);text(g,String.format("JAVA2D • %02d:%02d",frame/FPS/60,frame/FPS%60),1065,42,MUTED);}
    private static void label(Graphics2D g,String number,String title,String sub){g.setFont(new Font("Monospaced",Font.BOLD,18));text(g,number,54,104,GOLD);g.setFont(TITLE);text(g,title,94,115,INK);g.setFont(BODY);text(g,sub,96,148,MUTED);}
    private static void timeline(Graphics2D g,int frame){int y=692;g.setColor(GRID);g.fillRect(54,y,1172,3);g.setColor(CYAN);g.fillRect(54,y,(int)(1172.0*frame/(FRAMES-1)),3);}
    private static BufferedImage canvas(){return new BufferedImage(W,H,BufferedImage.TYPE_INT_RGB);}
    private static void setup(Graphics2D g){g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,RenderingHints.VALUE_TEXT_ANTIALIAS_ON);}
    private static void background(Graphics2D g){g.setPaint(new GradientPaint(0,0,new Color(255,249,236),W,H,new Color(250,230,214)));g.fillRect(0,0,W,H);for(int x=0;x<W;x+=64){g.setColor(new Color(139,100,31,12));g.drawLine(x,0,x,H);}for(int y=0;y<H;y+=64){g.setColor(new Color(139,100,31,12));g.drawLine(0,y,W,y);}}
    private static void drawMatrix(Graphics2D g,int[][] m,int x,int y,int size,int hiRow,int hiCol,boolean results){for(int i=0;i<m.length;i++)for(int j=0;j<m[i].length;j++){boolean hi=i==hiRow||j==hiCol;Color f=hi?(i==hiRow&&j==hiCol?new Color(246,237,200):new Color(227,236,223)):PANEL;cell(g,x+j*size,y+i*size,size,f,hi?GOLD:GRID);g.setFont(MONO);center(g,Integer.toString(m[i][j]),x+j*size+size/2,y+i*size+size/2+8,results&&hi?GOLD:INK);}brackets(g,x,y,m[0].length*size,m.length*size);}
    private static void brackets(Graphics2D g,int x,int y,int w,int h){g.setColor(MUTED);g.setStroke(new BasicStroke(3));g.draw(new Line2D.Float(x-12,y,x-12,y+h));g.draw(new Line2D.Float(x-12,y,x-3,y));g.draw(new Line2D.Float(x-12,y+h,x-3,y+h));g.draw(new Line2D.Float(x+w+12,y,x+w+12,y+h));g.draw(new Line2D.Float(x+w+3,y,x+w+12,y));g.draw(new Line2D.Float(x+w+3,y+h,x+w+12,y+h));}
    private static void cell(Graphics2D g,int x,int y,int size,Color fill,Color stroke){g.setColor(fill);g.fillRect(x+2,y+2,size-4,size-4);g.setColor(stroke);g.setStroke(new BasicStroke(1));g.drawRect(x+2,y+2,size-4,size-4);}
    private static void outline(Graphics2D g,int x,int y,int w,int h,Color c,int stroke){g.setColor(c);g.setStroke(new BasicStroke(stroke));g.drawRect(x,y,w,h);}
    private static void rounded(Graphics2D g,int x,int y,int w,int h,Color fill,Color stroke){g.setColor(fill);g.fillRoundRect(x,y,w,h,18,18);g.setColor(stroke);g.setStroke(new BasicStroke(2));g.drawRoundRect(x,y,w,h,18,18);}
    private static void symbol(Graphics2D g,String s,int x,int y){g.setFont(TITLE);center(g,s,x,y,GOLD);}
    private static void dot(Graphics2D g,int x,int y,Color c){g.setColor(c);g.fillOval(x-9,y-9,18,18);g.setColor(INK);g.setStroke(new BasicStroke(2));g.drawOval(x-9,y-9,18,18);}
    private static void line(Graphics2D g,int x1,int y1,int x2,int y2,Color c,int width){g.setColor(c);g.setStroke(new BasicStroke(width,BasicStroke.CAP_ROUND,BasicStroke.JOIN_ROUND));g.drawLine(x1,y1,x2,y2);}
    private static void text(Graphics2D g,String text,int x,int y,Color c){g.setColor(c);g.drawString(text,x,y);}
    private static void center(Graphics2D g,String text,int x,int y,Color c){FontMetrics fm=g.getFontMetrics();text(g,text,x-fm.stringWidth(text)/2,y,c);}
}
