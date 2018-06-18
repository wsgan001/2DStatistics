package experiments;

import utilities.ClassifierTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.core.elastic_distance_measures.ERPDistance;
import experiments.XY.DistanceType;

public class ERPTest {
	public static void main(String[] args) throws Exception {
		String[] datasets = DataSets.ucrNames;
		String dataDir = "G:/Êý¾Ý/TSC Problems/";
		Instances train, test;
		ERPDistance erp;
		kNN knn;
		int correct;
		double acc, err;

		StringBuilder st = new StringBuilder();
		System.out
				.println("Dataset \t ERP   \t 2DTbD-ERP");

		for (String dataset : datasets) {
//		for (int i=40;i<43;i++) {
//			String dataset=datasets[i];
			System.out.print(dataset + " \t ");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");

			// ERP
			erp = new ERPDistance(0.5,0.5);
			knn = new kNN(erp);
			correct = getCorrect(knn, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			// XY_ERP
			XY xy_erp = new XY(DistanceType.ERP);
			xy_erp.setAandB(0.5, 0.5);
			correct = getCorrect(xy_erp, train, test);
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
