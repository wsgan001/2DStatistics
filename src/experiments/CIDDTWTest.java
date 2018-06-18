package experiments;

import tsc_algorithms.NN_CID;
import utilities.ClassifierTools;
import utilities.fileIO.DataSets;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import experiments.XY.DistanceType;

public class CIDDTWTest {
	public static void main(String[] args) throws Exception {

		String[] datasets = DataSets.ucrNames;
		String dataDir = "G:/Êý¾Ý/TSC Problems/";
		Instances train, test, dTrain, dTest;
		kNN knn;
		int correct;
		double acc, err;


		System.out.println("Dataset  \t CIDDTW  \t 2DTbD-CIDDTW");;


		for (String dataset : datasets) {
			System.out.print(dataset + " \t ");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");
			

			

			// CID
			NN_CID k = new NN_CID();
			k.useDTW();
			correct = getCorrect(k, train, test);
			acc = (double) correct / test.numInstances();
			err = 1 - acc;
			System.out.print(err + " \t ");

			//XY_CID
			XY xy_cid = new XY(DistanceType.CIDDTW);
			xy_cid.setAandB(0.5, 0.5);
			correct = getCorrect(xy_cid, train, test);
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
