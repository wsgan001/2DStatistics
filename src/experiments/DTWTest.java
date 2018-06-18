package experiments;

import utilities.ClassifierTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;
import experiments.XY.DistanceType;

public class DTWTest {
	public static void main(String[] args) throws Exception {

		String[] datasets = DataSets.ucrNames;
		String dataDir = "G:/Êý¾Ý/TSC Problems/";
		Instances train, test, dTrain, dTest;
		kNN knn;
		int correct;
		double acc, err;

		
	

		StringBuilder st = new StringBuilder();
		System.out
				.println("Dataset  \t DTW  \t 2DTbD-DTW");


		for (String dataset : datasets) {

			System.out.print(dataset + " \t ");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");


			// DTW
			DTW_DistanceBasic dtw = new DTW_DistanceBasic();
			knn = new kNN(dtw);
			correct = getCorrect(knn, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// XY_DTW
			XY xy_dtw_f = new XY(DistanceType.DTW);
			xy_dtw_f.setAandB(0.5, 0.5);
			correct = getCorrect(xy_dtw_f, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			System.out.println();
		}

	}

	protected static int getCorrect(kNN knn, Instances train, Instances test)
			throws Exception {
		knn.buildClassifier(train);
		int correct = 0;
		for (int i = 0; i < test.numInstances(); i++) {
			if (test.instance(i).classValue() == knn.classifyInstance(test
					.instance(i))) {
				correct++;
			}
		}
		return correct;
	}
}
