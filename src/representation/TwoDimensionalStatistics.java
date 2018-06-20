package representation;

import weka.core.Instance;

import java.awt.geom.Line2D;
import java.awt.geom.Line2D.Double;

/**
 *
 * @author jc
 * 
 * 
 */

public class TwoDimensionalStatistics{
	public double[][] tsTo2D(Instance  timeSeries){
		int length=timeSeries.numAttributes();
		double angle= 2*Math.PI/length;
		double[][] representation2D=new double[2][length-1];
		for(int i=0;i<length;i++){
			double value=timeSeries.value(i);
			representation2D[0][i]=value*Math.cos(angle*i);
			representation2D[1][i]=value*Math.sin(angle*i);
		}
		return representation2D;
	}
	
	public boolean visible(double[][] representation2D, int begin, int end){
		if(end-begin==1){
			return true;
		}
		int length=representation2D[0].length;
		if((begin==0)&&(end==length-1)){
			return true;
		}
		
		double x1=representation2D[0][begin];
		double x2=representation2D[0][end];
		double y1=representation2D[1][begin];
		double y2=representation2D[1][end];
		
		Line2D.Double line=new Line2D.Double(x1,y1,x2,y2);
		
		for(int i=begin+1;i<end-1;i++){
			
			if(line.intersectsLine(representation2D[0][i], representation2D[1][i], representation2D[0][i+1], representation2D[1][i+1])){
				return false;
			}
		}
		for(int i=end+1;i<length-1;i++){
			if(line.intersectsLine(representation2D[0][i], representation2D[1][i], representation2D[0][i+1], representation2D[1][i+1])){
				return false;
			}
		}
		
		for(int i=0;i<begin-1;i++){
			if(line.intersectsLine(representation2D[0][i], representation2D[1][i], representation2D[0][i+1], representation2D[1][i+1])){
				return false;
			}
		}
		if(end-begin==2){
			if(line.ptSegDist(representation2D[0][begin+1], representation2D[1][begin+1])==0){
				return false;
			}
		}
		if((begin==0)&&(end==length-2)){
			if(line.ptSegDist(representation2D[0][end+1], representation2D[1][end+1])==0){
				return false;
			}
		}
		if((begin==1)&&(end==length-1)){
			if(line.ptSegDist(representation2D[0][0], representation2D[1][0])==0){
				return false;
			}
		}
		
		return true;
	}
	
	
	public static void main(String[] args) throws Exception {
		// todo
		
		double[] x={0.3,1.2,2,2,2,0.8,0.7,0};
		double[] y={0,0.1,-0.1,1.3,2.0,2.1,0.9,1.1};
		double[][] representation2D=new double[2][];
		representation2D[0]=x;
		representation2D[1]=y;
		TwoDimensionalStatistics ds=new TwoDimensionalStatistics();
		for(int i=0;i<7;i++){
			for(int j=i+1;j<=7;j++){
				if(ds.visible(representation2D, i, j)){
					System.out.println(i+"\t"+j+"\t可见");
				}else{
					System.out.println(i+"\t"+j+"\t不可见");
				}
			}
		}
		
	}

}
