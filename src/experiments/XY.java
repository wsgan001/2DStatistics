package experiments;

import representation.xy.XYFilter;
import tsc_algorithms.DD_DTW;
import tsc_algorithms.DD_DTW.DistanceType;
import tsc_algorithms.NN_CID.CIDDistance;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.DTW;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.elastic_distance_measures.DTW_DistanceEfficient;
import weka.core.elastic_distance_measures.ERPDistance;
import weka.core.elastic_distance_measures.MSMDistance;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author jc
 * 
 * 
 */

public class XY extends kNN {

	protected XYEuclideanDistance distanceFunction;
	protected boolean paramsSet;
	protected boolean sampleForCV = false;
	protected double prop;

	public enum DistanceType {
		EUCLIDEAN, DTW, CID, CIDDTW,ERP,MSM
	};

	// defaults to Euclidean distance
	public XY() {
		super();
		this.distanceFunction = new XYDTW();
		this.paramsSet = false;

	}

	public XY(DistanceType distType) {
		super();
		if (distType == DistanceType.EUCLIDEAN) {
			this.distanceFunction = new XYEuclideanDistance();
		} else if (distType == DistanceType.DTW) {
			this.distanceFunction = new XYDTW();
		} else if (distType == DistanceType.CID) {
			this.distanceFunction = new XYCID();
		} else if (distType == DistanceType.CIDDTW) {
			this.distanceFunction = new XYCIDDTW();
		} else if (distType == DistanceType.ERP) {
			this.distanceFunction = new XYERP();
		}else {
			this.distanceFunction = new XYMSM();
		}
		this.paramsSet = false;
	}

	public void setAandB(double a, double b) {
		this.distanceFunction.a = a;
		this.distanceFunction.b = b;
		this.paramsSet = true;
	}

	@Override
	public void buildClassifier(Instances train) {
		if (!paramsSet) {
			this.distanceFunction.crossValidateForAandB(train);
			paramsSet = true;
		}
		this.setDistanceFunction(this.distanceFunction);
		super.buildClassifier(train);
	}

	public static class XYEuclideanDistance extends EuclideanDistance {

		protected double a;
		protected double b;
		public boolean sampleTrain = true; // Change back to default to false

		public XYEuclideanDistance() {
			this.a = 1;
			this.b = 0;
			// defaults to no derivative input
		}

		public XYEuclideanDistance(Instances train) {
			this.crossValidateForAandB(train);
		}

		public XYEuclideanDistance(double a, double b) {
			this.a = a;
			this.b = b;
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {
			// double dist = 0;
			double distX = 0;
			double distY = 0;

			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			int length = first.numAttributes() - classPenalty;
			double angle = 2 * Math.PI / length;
			for (int i = 0; i < length; i++) {
				double value1 = first.value(i);
				double value2 = second.value(i);

				double thisAngle = angle * i;
				// dist += Math.pow((value1 - value2),2) ;
				distX += Math.pow((value1 - value2) * Math.cos(thisAngle), 2);
				distY += Math.pow((value1 - value2) * Math.sin(thisAngle), 2);

			}
			return (a * Math.sqrt(distX) + b * Math.sqrt(distY));
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {
			// double dist = 0;
			double distX = 0;
			double distY = 0;

			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			int length = first.numAttributes() - classPenalty;
			double angle = 2 * Math.PI / length;
			for (int i = 0; i < length; i++) {
				double value1 = first.value(i);
				double value2 = second.value(i);

				double thisAngle = angle * i;
				// dist += Math.pow((value1 - value2),2) ;
				distX += Math.pow((value1 - value2) * Math.cos(thisAngle), 2);
				distY += Math.pow((value1 - value2) * Math.sin(thisAngle), 2);

			}
			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

		// changed to now return the predictions of the best alpha parameter
		public double[] crossValidateForAandB(Instances tr) {
			Instances train = tr;
			if (sampleTrain) {
				tr = InstanceTools.subSample(tr, tr.numInstances() / 10, 0);
			}

			double[] labels = new double[train.numInstances()];
			for (int i = 0; i < train.numInstances(); i++) {
				labels[i] = train.instance(i).classValue();
			}

			double[] a = new double[101];
			double[] b = new double[101];

			for (int alphaId = 0; alphaId <= 100; alphaId++) {
				a[alphaId] = (100.0 - alphaId) / 100;
				b[alphaId] = alphaId * 1.0 / 100;
			}

			int n = train.numInstances();
			int k = a.length;
			int[] mistakes = new int[k];

			double[] D;
			double[] L;
			double[] d;
			double dist;
			double dDist;

			double[][] LforAll = new double[n][];

			double[] individualDistances;

			for (int i = 0; i < n; i++) {

				D = new double[k];
				L = new double[k];
				for (int j = 0; j < k; j++) {
					D[j] = Double.MAX_VALUE;
				}

				for (int j = 0; j < n; j++) {
					if (i == j) {
						continue;
					}

					individualDistances = this.getNonScaledDistances(
							train.instance(i), train.instance(j));
					dist = individualDistances[0];
					dDist = individualDistances[1];

					d = new double[k];

					for (int alphaId = 0; alphaId < k; alphaId++) {
						d[alphaId] = a[alphaId] * dist + b[alphaId] * dDist;
						if (d[alphaId] < D[alphaId]) {
							D[alphaId] = d[alphaId];
							L[alphaId] = labels[j];
						}
					}
				}

				for (int alphaId = 0; alphaId < k; alphaId++) {
					if (L[alphaId] != labels[i]) {
						mistakes[alphaId]++;
					}
				}
				LforAll[i] = L;
			}

			int bsfMistakes = Integer.MAX_VALUE;
			int bsfAlphaId = -1;
			for (int alpha = 0; alpha < k; alpha++) {
				if (mistakes[alpha] < bsfMistakes) {
					bsfMistakes = mistakes[alpha];
					bsfAlphaId = alpha;
				}
			}

			this.a = a[bsfAlphaId];
			this.b = b[bsfAlphaId];
			double[] bestAlphaPredictions = new double[train.numInstances()];
			for (int i = 0; i < bestAlphaPredictions.length; i++) {
				bestAlphaPredictions[i] = LforAll[i][bsfAlphaId];
			}
			// System.out.println("a:"+this.a+"b:"+this.b);
			return bestAlphaPredictions;
		}

		public double getA() {
			return a;
		}

		public double getB() {
			return b;
		}

	}

	public static class XYDTW extends XYEuclideanDistance {

		public XYDTW() {
			super();
		}

		public XYDTW(Instances train) {
			super(train);
		}

		public XYDTW(double a, double b) {
			super(a, b);
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double[] distances = getNonScaledDistances(first, second);
			return a * distances[0] + b * distances[1];
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {

			double distX = 0;
			double distY = 0;

			// DTW dtw = new DTW();
			DTW_DistanceBasic dtw = new DTW_DistanceBasic();
			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			XYFilter filter = new XYFilter();
			Instances tempX = new Instances(first.dataset(), 0);
			Instances tempY = new Instances(first.dataset(), 0);
			tempX.add(first);
			tempX.add(second);
			tempY.add(first);
			tempY.add(second);
			try {
				tempX = filter.processX(tempX);
				tempY = filter.processY(tempY);
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}

			distX = dtw.distance(tempX.get(0), tempX.get(1));
			distY = dtw.distance(tempY.get(0), tempY.get(1), Double.MAX_VALUE);

			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

	}

	public static class XYCID extends XYEuclideanDistance {

		public XYCID() {
			super();
		}

		public XYCID(Instances train) {
			super(train);
		}

		public XYCID(double a, double b) {
			super(a, b);
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double[] distances = getNonScaledDistances(first, second);
			return a * distances[0] + b * distances[1];
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {

			double distX = 0;
			double distY = 0;

			// DTW dtw = new DTW();
			DTW_DistanceBasic dtw = new DTW_DistanceBasic();
			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			XYFilter filter = new XYFilter();
			Instances tempX = new Instances(first.dataset(), 0);
			Instances tempY = new Instances(first.dataset(), 0);
			tempX.add(first);
			tempX.add(second);
			tempY.add(first);
			tempY.add(second);
			try {
				tempX = filter.processX(tempX);
				tempY = filter.processY(tempY);
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}

			CIDDistance cd = new CIDDistance();

			distX = cd.distance(tempX.get(0), tempX.get(1));
			distY = cd.distance(tempY.get(0), tempY.get(1), Double.MAX_VALUE);

			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

	}

	public static class CIDDistance extends EuclideanDistance {

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double d = 0;
			// Find the acf terms
			double d1 = 0, d2 = 0;
			double[] data1 = first.toDoubleArray();
			double[] data2 = second.toDoubleArray();
			for (int i = 0; i < first.numAttributes() - 1; i++)
				d += (data1[i] - data2[i]) * (data1[i] - data2[i]);
			d = Math.sqrt(d);
			for (int i = 0; i < first.numAttributes() - 2; i++)
				d1 += (data1[i] - data1[i + 1]) * (data1[i] - data1[i + 1]);
			for (int i = 0; i < first.numAttributes() - 2; i++)
				d2 += (data2[i] - data2[i + 1]) * (data2[i] - data2[i + 1]);
			d1 = Math.sqrt(d1 + 0.001); // This is from theircode
			d2 = Math.sqrt(d2 + 0.001); // This is from theircode
			if (d1 < d2) {
				double temp = d1;
				d1 = d2;
				d2 = temp;
			}
			d = Math.sqrt(d);
			d = d * (d1 / d2);
			return d;
		}

	}

	public static class XYCIDDTW extends XYEuclideanDistance {

		public XYCIDDTW() {
			super();
		}

		public XYCIDDTW(Instances train) {
			super(train);
		}

		public XYCIDDTW(double a, double b) {
			super(a, b);
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double[] distances = getNonScaledDistances(first, second);
			return a * distances[0] + b * distances[1];
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {

			double distX = 0;
			double distY = 0;
			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			XYFilter filter = new XYFilter();
			Instances tempX = new Instances(first.dataset(), 0);
			Instances tempY = new Instances(first.dataset(), 0);
			tempX.add(first);
			tempX.add(second);
			tempY.add(first);
			tempY.add(second);
			try {
				tempX = filter.processX(tempX);
				tempY = filter.processY(tempY);
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}

			CIDDTWDistance cd = new CIDDTWDistance();

			distX = cd.distance(tempX.get(0), tempX.get(1));
			distY = cd.distance(tempY.get(0), tempY.get(1), Double.MAX_VALUE);

			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

	}

	public static class CIDDTWDistance extends CIDDistance {
		DTW_DistanceEfficient dtw = new DTW_DistanceEfficient();

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double d = 0;
			// Find the acf terms
			double d1 = 0, d2 = 0;
			double[] data1 = first.toDoubleArray();
			double[] data2 = second.toDoubleArray();

			d = dtw.distance(first, second);
			for (int i = 0; i < first.numAttributes() - 2; i++)
				d1 += (data1[i] - data1[i + 1]) * (data1[i] - data1[i + 1]);
			for (int i = 0; i < first.numAttributes() - 2; i++)
				d2 += (data2[i] - data2[i + 1]) * (data2[i] - data2[i + 1]);
			d1 = Math.sqrt(d1) + 0.001; // This is from theircode
			d2 = Math.sqrt(d2) + 0.001; // This is from theircode
			if (d1 < d2) {
				double temp = d1;
				d1 = d2;
				d2 = temp;
			}
			d = d * (d1 / d2);
			return d;
		}

	}
	
	public static class XYERP extends XYEuclideanDistance {

		public XYERP() {
			super();
		}

		public XYERP(Instances train) {
			super(train);
		}

		public XYERP(double a, double b) {
			super(a, b);
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double[] distances = getNonScaledDistances(first, second);
			return a * distances[0] + b * distances[1];
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {

			double distX = 0;
			double distY = 0;

			
			ERPDistance erp = new ERPDistance(0.5,0.5);
			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			XYFilter filter = new XYFilter();
			Instances tempX = new Instances(first.dataset(), 0);
			Instances tempY = new Instances(first.dataset(), 0);
			tempX.add(first);
			tempX.add(second);
			tempY.add(first);
			tempY.add(second);
			try {
				tempX = filter.processX(tempX);
				tempY = filter.processY(tempY);
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}

			

			distX = erp.distance(tempX.get(0), tempX.get(1));
			distY = erp.distance(tempY.get(0), tempY.get(1), Double.MAX_VALUE);

			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

	}
	
	
	public static class XYMSM extends XYEuclideanDistance {

		public XYMSM() {
			super();
		}

		public XYMSM(Instances train) {
			super(train);
		}

		public XYMSM(double a, double b) {
			super(a, b);
		}

		@Override
		public double distance(Instance one, Instance two) {
			return this.distance(one, two, Double.MAX_VALUE);
		}

		@Override
		public double distance(Instance one, Instance two, double cutoff,
				PerformanceStats stats) {
			return this.distance(one, two, cutoff);
		}

		@Override
		public double distance(Instance first, Instance second, double cutoff) {

			double[] distances = getNonScaledDistances(first, second);
			return a * distances[0] + b * distances[1];
		}

		public double[] getNonScaledDistances(Instance first, Instance second) {

			double distX = 0;
			double distY = 0;

			
			MSMDistance msm=new MSMDistance();
			int classPenalty = 0;
			if (first.classIndex() > 0) {
				classPenalty = 1;
			}

			XYFilter filter = new XYFilter();
			Instances tempX = new Instances(first.dataset(), 0);
			Instances tempY = new Instances(first.dataset(), 0);
			tempX.add(first);
			tempX.add(second);
			tempY.add(first);
			tempY.add(second);
			try {
				tempX = filter.processX(tempX);
				tempY = filter.processY(tempY);
			} catch (Exception e) {
				e.printStackTrace();
				return null;
			}

			

			distX = msm.distance(tempX.get(0), tempX.get(1));
			distY = msm.distance(tempY.get(0), tempY.get(1), Double.MAX_VALUE);

			return new double[] { Math.sqrt(distX), Math.sqrt(distY) };
		}

	}
	

}
