/*
     * copyright: Anthony Bagnall
 * */
package weka.classifiers.trees.shapelet_trees;

//note: need to implement some way of making a binary decision about a shapelet
// done: using infomation gain after the shapelet has been selected with kruskal-wallis
import weka.core.*;
import weka.core.Instances;
import java.util.TreeMap;
import java.util.ArrayList;
import java.util.Collections;
import java.io.FileReader;

import java.io.FileWriter;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

//import java.io.File;
//import java.util.Scanner;
public class KruskalWallisTree extends AbstractClassifier{

    private ShapeletNode root;
    private String logFileName;
    private int minLength, maxLength;
    
    public KruskalWallisTree(String logFileName) throws Exception {
        this.root = new ShapeletNode();
        this.logFileName = logFileName;
        FileWriter fw = new FileWriter(logFileName);
        fw.close();
    }

    public void setShapeletMinMaxLength(int minLength, int maxLength){
        this.minLength = minLength;
        this.maxLength = maxLength;
    }
       
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(minLength < 1 || maxLength < 1){
            throw new Exception("Shapelet minimum or maximum length is incorrectly specified!");
        }
        
        root.initialiseNode(data, minLength, maxLength, 0);
    }

    @Override
    public double classifyInstance(Instance instance) {
        return root.classifyInstance(instance);
    }

    private Shapelet getRootShapelet() {
        return this.root.shapelet;
    }

    private class ShapeletNode {

        private ShapeletNode leftNode;
        private ShapeletNode rightNode;
        private double classDecision;
        private Shapelet shapelet;

        public ShapeletNode() {
            leftNode = null;
            rightNode = null;
            classDecision = -1;
        }

        public void initialiseNode(Instances data, int minShapeletLength, int maxShapeletLength, int level) throws Exception {
            FileWriter fw = new FileWriter(logFileName, true);
            fw.append("level:" + level + ", numInstances:" + data.numInstances() + "\n");
            fw.close();

            // 1. check whether this is a leaf node with only one class present
            double firstClassValue = data.instance(0).classValue();
            boolean oneClass = true;
            for (int i = 1; i < data.numInstances(); i++) {
                if (data.instance(i).classValue() != firstClassValue) {
                    oneClass = false;
                    break;
                }
            }

            if (oneClass == true) {
                this.classDecision = firstClassValue; // no need to find shapelet, base case
//                                System.out.println("base case");
                fw = new FileWriter(logFileName, true);
                fw.append("class decision here: " + firstClassValue + "\n");
                fw.close();
            } else { // recursively call method to create left and right children nodes
                try {
                    // 1. find the best shapelet to split the data
                    this.shapelet = findBestShapelet(data, minShapeletLength, maxShapeletLength);

                    // 2. split the data using the shapelet and create new data sets
                    double dist;
//                                System.out.println("Threshold:"+shapelet.getThreshold());
//                                System.out.println("length:"+shapelet.getLength());
                    ArrayList<Instance> splitLeft = new ArrayList<Instance>();
                    ArrayList<Instance> splitRight = new ArrayList<Instance>();

                    for (int i = 0; i < data.numInstances(); i++) {
                        dist = subsequenceDistance(this.shapelet.content, data.instance(i).toDoubleArray());
//                                System.out.println("dist:"+dist);
                        if (dist < shapelet.splitThreshold) {
                            splitLeft.add(data.instance(i));
//                                                System.out.println("gone left");
                        } else {
                            splitRight.add(data.instance(i));
//                                                System.out.println("gone right");
                        }
                    }

                    // write to file here!!!!
                    fw = new FileWriter(logFileName, true);
                    fw.append("seriesId, startPos, length, kwstat, splitThresh\n");
                    fw.append(this.shapelet.seriesId + "," + this.shapelet.startPos + "," + this.shapelet.content.length + "," + this.shapelet.getKruskalWallisStat() + "," + this.shapelet.splitThreshold + "\n");
                    for (int j = 0; j < this.shapelet.content.length; j++) {
                        fw.append(this.shapelet.content[j] + ",");
                    }
                    fw.append("\n");
                    fw.close();

                    System.out.println("shapelet completed at:" + System.nanoTime());


//                        System.out.println("leftSize:"+splitLeft.size());
//                        System.out.println("leftRight:"+splitRight.size());

                    // 5. initialise and recursively compute children nodes
                    leftNode = new ShapeletNode();
                    rightNode = new ShapeletNode();
//                                System.out.println("SplitLeft:");

                    Instances leftInstances = new Instances(data, splitLeft.size());
                    for (int i = 0; i < splitLeft.size(); i++) {
                        leftInstances.add(splitLeft.get(i));
                    }
                    Instances rightInstances = new Instances(data, splitRight.size());
                    for (int i = 0; i < splitRight.size(); i++) {
                        rightInstances.add(splitRight.get(i));
                    }

                    fw = new FileWriter(logFileName, true);
                    fw.append("left size under level " + level + ": " + leftInstances.numInstances() + "\n");
                    fw.close();
                    System.out.println("left: " + leftInstances.numInstances());
                    System.out.println("right: " + rightInstances.numInstances());

                    System.out.println("split: " + this.shapelet.startPos);

                    leftNode.initialiseNode(leftInstances, minShapeletLength, maxShapeletLength, (level + 1));
//                                System.out.println("SplitRight:");

                    fw = new FileWriter(logFileName, true);
                    fw.append("right size under level " + level + ": " + rightInstances.numInstances() + "\n");
                    fw.close();

                    rightNode.initialiseNode(rightInstances, minShapeletLength, maxShapeletLength, (level + 1));
                } catch (Exception e) {
                    System.out.println("Problem initialising tree node: " + e);
                    e.printStackTrace();
                }
            }
        }

        public double classifyInstance(Instance instance) {
            if (this.leftNode == null) {
                return this.classDecision;
            } else {
                double distance;
                distance = subsequenceDistance(this.shapelet.content, instance);

                if (distance < this.shapelet.splitThreshold) {
                    return leftNode.classifyInstance(instance);
                } else {
                    return rightNode.classifyInstance(instance);
                }
            }
        }
    }

    //#
    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {
        long startTime = System.nanoTime();
        this.findBestShapelet(data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double)(finishTime - startTime) / 1000000000.0;
    }

    // edited from findBestKShapeletsCached
    private Shapelet findBestShapelet(Instances data, int minShapeletLength, int maxShapeletLength) {

        Shapelet bestShapelet = null;


        TreeMap<Double, Integer> classDistributions = getClassDistributions(data); // used to calc info gain

        //for all time series
        System.out.println("Processing data: ");
        for (int i = 0; i < data.numInstances(); i++) {
//                System.out.println((1+i)+"/"+data.numInstances()+"\t Started: "+getTime());

            double[] wholeCandidate = data.instance(i).toDoubleArray();
            // for all lengths
            for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
                //for all possible starting positions of that length
                for (int start = 0; start <= wholeCandidate.length - length - 1; start++) { //-1 = avoid classVal - handle later for series with no class val
                    // CANDIDATE ESTABLISHED - got original series, length and starting position
                    // extract relevant part into a double[] for processing
                    double[] candidate = new double[length];
                    for (int m = start; m < start + length; m++) {
                        candidate[m - start] = wholeCandidate[m];
                    }

                    candidate = zNorm(candidate, false);
                    Shapelet candidateShapelet = checkCandidate(candidate, data, i, start, classDistributions);

                    if (bestShapelet == null || candidateShapelet.compareTo(bestShapelet) < 0) {
                        bestShapelet = candidateShapelet;
                    }
                }
            }
        }

        bestShapelet.calculateBestSplitPoint(classDistributions);

        return bestShapelet;
    }

    /**
     *
     * @param shapelets the input Shapelets to remove self similar Shapelet objects from
     * @return a copy of the input ArrayList with self-similar shapelets removed
     */
    private static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets) {
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<Shapelet> outputShapelets = new ArrayList<Shapelet>();
        boolean[] selfSimilar = new boolean[shapelets.size()];

        // to keep tract of self similarity - assume nothing is similar to begin with
        for (int i = 0; i < shapelets.size(); i++) {
            selfSimilar[i] = false;
        }

        for (int i = 0; i < shapelets.size(); i++) {
            if (selfSimilar[i] == false) {
                outputShapelets.add(shapelets.get(i));
                for (int j = i + 1; j < shapelets.size(); j++) {
                    if (selfSimilar[j] == false && selfSimilarity(shapelets.get(i), shapelets.get(j))) { // no point recalc'ing if already self similar to something
                        selfSimilar[j] = true;
                    }
                }
            }
        }
        return outputShapelets;
    }

    /**
     *
     * @param k the maximum number of shapelets to be returned after combining the two lists
     * @param kBestSoFar the (up to) k best shapelets that have been observed so far, passed in to combine with shapelets from a new series
     * @param timeSeriesShapelets the shapelets taken from a new series that are to be merged in descending order of fitness with the kBestSoFar
     * @return an ordered ArrayList of the best k (or less) Shapelet objects from the union of the input ArrayLists
     */
    private ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets) {

        ArrayList<Shapelet> newBestSoFar = new ArrayList<Shapelet>();
        for (int i = 0; i < timeSeriesShapelets.size(); i++) {
            kBestSoFar.add(timeSeriesShapelets.get(i));
        }
        Collections.sort(kBestSoFar);
        if (kBestSoFar.size() < k) {
            return kBestSoFar; // no need to return up to k, as there are not k shapelets yet
        }
        for (int i = 0; i < k; i++) {
            newBestSoFar.add(kBestSoFar.get(i));
        }

        return newBestSoFar;
    }

    /**
     *
     * @param data the input data set that the class distributions are to be derived from
     * @return a TreeMap<Double, Integer> in the form of <Class Value, Frequency>
     */
    private static TreeMap<Double, Integer> getClassDistributions(Instances data) {
        TreeMap<Double, Integer> classDistribution = new TreeMap<Double, Integer>();
        double classValue;
        for (int i = 0; i < data.numInstances(); i++) {
            classValue = data.instance(i).classValue();
            boolean classExists = false;
            for (Double d : classDistribution.keySet()) {
                if (d == classValue) {
                    int temp = classDistribution.get(d);
                    temp++;
                    classDistribution.put(classValue, temp);
                    classExists = true;
                }
            }
            if (classExists == false) {
                classDistribution.put(classValue, 1);
            }
        }
        return classDistribution;
    }

    /**
     *
     * @param candidate the data from the candidate Shapelet
     * @param data the entire data set to compare the candidate to
     * @param data the entire data set to compare the candidate to
     * @return a TreeMap<Double, Integer> in the form of <Class Value, Frequency>
     */
    private static Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, TreeMap classDistribution) {

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<OrderLineObj>();

        for (int i = 0; i < data.numInstances(); i++) {
            double distance = subsequenceDistance(candidate, data.instance(i));
            double classVal = data.instance(i).classValue();

            boolean added = false;
            // add to orderline
            if (orderline.isEmpty()) {
                orderline.add(new OrderLineObj(distance, classVal));
                added = true;
            } else {
                for (int j = 0; j < orderline.size(); j++) {
                    if (added == false && orderline.get(j).distance > distance) {
                        orderline.add(j, new OrderLineObj(distance, classVal));
                        added = true;
                    }
                }
            }
            // if obj hasn't been added, must be furthest so add at end
            if (added == false) {
                orderline.add(new OrderLineObj(distance, classVal));
            }
        }
        // create a shapelet object to store all necessary info, i.e.
        // content, seriesId, then calc info gain, plit threshold and separation gap
        Shapelet shapelet = new Shapelet(candidate, seriesId, startPos);
        shapelet.calculateKruskalWallis(orderline);
//        shapelet.calcInfoGainAndThreshold(orderline, classDistribution);

        // note: early abandon entropy pruning would appear here, but has been ommitted
        // in favour of a clear multi-class information gain calculation. Could be added in
        // this method in the future for speed up, but distance early abandon is more important

        return shapelet;
    }

    /**
     *
     * @param candidate
     * @param timeSeriesIns
     * @return
     */
    public static double subsequenceDistance(double[] candidate, Instance timeSeriesIns) {
        double[] timeSeries = timeSeriesIns.toDoubleArray();
        return subsequenceDistance(candidate, timeSeries);
    }

    public static double subsequenceDistance(double[] candidate, double[] timeSeries) {

//        double[] timeSeries = timeSeriesIns.toDoubleArray();
        double bestSum = Double.MAX_VALUE;
        double sum = 0;
        double[] subseq;

        // for all possible subsequences of two
        for (int i = 0; i <= timeSeries.length - candidate.length - 1; i++) {
            sum = 0;
            // get subsequence of two that is the same lenght as one
            subseq = new double[candidate.length];

            for (int j = i; j < i + candidate.length; j++) {
                subseq[j - i] = timeSeries[j];
            }
            subseq = zNorm(subseq, false); // Z-NORM HERE
            for (int j = 0; j < candidate.length; j++) {
                sum += (candidate[j] - subseq[j]) * (candidate[j] - subseq[j]);
            }
            if (sum < bestSum) {
                bestSum = sum;
            }
        }
        return (1.0 / candidate.length * bestSum);
    }

    /**
     *
     * @param input
     * @param classValOn
     * @return
     */
    public static double[] zNorm(double[] input, boolean classValOn) {
        double mean;
        double stdv;

        double classValPenalty = 0;
        if (classValOn) {
            classValPenalty = 1;
        }
        double[] output = new double[input.length];
        double seriesTotal = 0;

        for (int i = 0; i < input.length - classValPenalty; i++) {
            seriesTotal += input[i];
        }

        mean = seriesTotal / (input.length - classValPenalty);
        stdv = 0;
        for (int i = 0; i < input.length - classValPenalty; i++) {
            stdv += (input[i] - mean) * (input[i] - mean);
        }

        stdv = stdv / input.length - classValPenalty;
        stdv = Math.sqrt(stdv);

        for (int i = 0; i < input.length - classValPenalty; i++) {
            output[i] = (input[i] - mean) / stdv;
        }

        if (classValOn == true) {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    /**
     *
     * @param fileName
     * @return
     */
    public static Instances loadData(String fileName) {
        Instances data = null;
        try {
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println(" Error =" + e + " in method loadData");
        }
        return data;
    }

    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate) {
        if (candidate.seriesId == shapelet.seriesId) {
            if (candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.content.length) { //candidate starts within exisiting shapelet
                return true;
            }
            if (shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.content.length) {
                return true;
            }
        }
        return false;
    }

    private static class Shapelet implements Comparable<Shapelet> {

        private double[] content;
        private int seriesId;
        private int startPos;
        // for purpose of finding split point
        private ArrayList<OrderLineObj> orderline;
        private double kruskalWallaceStat;
        private double splitThreshold;
        private double separationGap;

        private Shapelet(double[] content, int seriesId, int startPos) {
            this.content = content;
            this.seriesId = seriesId;
            this.startPos = startPos;
            this.orderline = null;
        }

        private Shapelet(double[] content) {
            this.content = content;
        }

        private void calculateKruskalWallis(ArrayList<OrderLineObj> orderline) {

            // sort and assign ranks
            Collections.sort(orderline);
            assignRanks(orderline);

            // 2) sum of ranks for each class and mean ranks for each class
            TreeMap<Double, Double> classRankSums = new TreeMap<Double, Double>();
            TreeMap<Double, Integer> classCounts = new TreeMap<Double, Integer>();
            TreeMap<Double, Double> classRankMeans = new TreeMap<Double, Double>();

            double classVal, thisRank, oldRank;
            int oldCount;
            for (int i = 0; i < orderline.size(); i++) {
                classVal = orderline.get(i).classVal;
                thisRank = orderline.get(i).rank;

                if (classRankSums.containsKey(classVal)) {
                    oldRank = classRankSums.get(classVal);
                    oldCount = classCounts.get(classVal);

                    classRankSums.put(classVal, oldRank + thisRank);
                    classCounts.put(classVal, oldCount + 1);
                } else {
                    classRankSums.put(classVal, thisRank);
                    classCounts.put(classVal, 1);
                }
            }

            double thisMean;
            for (Double d : classRankSums.keySet()) {
                thisMean = classRankSums.get(d) / classCounts.get(d);
                classRankMeans.put(d, thisMean);
            }

            //3) Overall mean rank
            double overallMeanRank = (1.0 + orderline.size()) / 2;

            //4) sum of squared deviations from the overall mean rank
            double s = 0;

            for (Double d : classRankMeans.keySet()) {
                s += classCounts.get(d) * (classRankMeans.get(d) - overallMeanRank) * (classRankMeans.get(d) - overallMeanRank);
            }

            //5) weight s with the scale factor
            double h = 12.0 / (orderline.size() * (orderline.size() + 1)) * s;

//            return h;

            this.orderline = orderline;
            this.kruskalWallaceStat = h;
        }

        public double getKruskalWallisStat() {
            return this.kruskalWallaceStat;
        }

        public int getLength() {
            return this.content.length;
        }

        // finds the best split point by calulating the info gain of all possible splits
        // note: still better than raw info gain as we can just call this ONCE when the best shapelets is found with kruskal-wallis
        //
        // WHY I THINK THIS IS OK
        // We're not saying that info gain is bad, we're just saying it's ott. Therefore by doing this, we find the best shapelet through less-
        // intensive alternative means, and then find the best split point in that data using this selected shapelet and calcing the info gain (like a tree would!)
        private void calculateBestSplitPoint(TreeMap<Double, Integer> classDistribution) {
            double lastDist = orderline.get(0).distance;
            double thisDist = -1;

            double bsfGain = -1;
            double threshold = -1;

            for (int i = 1; i < orderline.size(); i++) {
                thisDist = orderline.get(i).distance;
                if (i == 1 || thisDist != lastDist) { // check that threshold has moved(no point in sampling identical thresholds)- special case - if 0 and 1 are the same dist

                    // count class instances below and above threshold
                    TreeMap<Double, Integer> lessClasses = new TreeMap<Double, Integer>();
                    TreeMap<Double, Integer> greaterClasses = new TreeMap<Double, Integer>();

                    for (double j : classDistribution.keySet()) {
                        lessClasses.put(j, 0);
                        greaterClasses.put(j, 0);
                    }

                    int sumOfLessClasses = 0;
                    int sumOfGreaterClasses = 0;

                    //visit those below threshold
                    for (int j = 0; j < i; j++) {
                        double thisClassVal = orderline.get(j).classVal;
                        int storedTotal = lessClasses.get(thisClassVal);
                        storedTotal++;
                        lessClasses.put(thisClassVal, storedTotal);
                        sumOfLessClasses++;
                    }

                    //visit those above threshold
                    for (int j = i; j < orderline.size(); j++) {
                        double thisClassVal = orderline.get(j).classVal;
                        int storedTotal = greaterClasses.get(thisClassVal);
                        storedTotal++;
                        greaterClasses.put(thisClassVal, storedTotal);
                        sumOfGreaterClasses++;
                    }

                    int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                    double parentEntropy = entropy(classDistribution);

                    // calculate the info gain below the threshold
                    double lessFrac = (double) sumOfLessClasses / sumOfAllClasses;
                    double entropyLess = entropy(lessClasses);
                    // calculate the info gain above the threshold
                    double greaterFrac = (double) sumOfGreaterClasses / sumOfAllClasses;
                    double entropyGreater = entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;

                    if (gain > bsfGain) {
                        bsfGain = gain;
                        threshold = (thisDist - lastDist) / 2 + lastDist;
                    }
                }
                lastDist = thisDist;
            }

            this.splitThreshold = threshold;
            this.separationGap = calculateSeparationGap(orderline, threshold);

        }

        private double calculateSeparationGap(ArrayList<OrderLineObj> orderline, double distanceThreshold) {

            double sumLeft = 0;
            double leftSize = 0;
            double sumRight = 0;
            double rightSize = 0;

            for (int i = 0; i < orderline.size(); i++) {
                if (orderline.get(i).distance < distanceThreshold) {
                    sumLeft += orderline.get(i).distance;
                    leftSize++;
                } else {
                    sumRight += orderline.get(i).distance;
                    rightSize++;
                }
            }

            double thisSeparationGap = 1 / rightSize * sumRight - 1 / leftSize * sumLeft; //!!!! they don't divide by 1 in orderLine::minGap(int j)

            if (rightSize == 0 || leftSize == 0) {
                return -1; // obviously there was no seperation, which is likely to be very rare but i still caused it!
            }                //e.g if all data starts with 0, first shapelet length =1, there will be no seperation as all time series are same dist
            // equally true if all data contains the shapelet candidate, which is a more realistic example

            return thisSeparationGap;
        }

        private static double entropy(TreeMap<Double, Integer> classDistributions) {
            if (classDistributions.size() == 1) {
                return 0;
            }

            double thisPart;
            double toAdd;
            int total = 0;
            for (Double d : classDistributions.keySet()) {
                total += classDistributions.get(d);
            }
            // to avoid NaN calculations, the individual parts of the entropy are calculated and summed.
            // i.e. if there is 0 of a class, then that part would calculate as NaN, but this can be caught and
            // set to 0.
            ArrayList<Double> entropyParts = new ArrayList<Double>();
            for (Double d : classDistributions.keySet()) {
                thisPart = (double) classDistributions.get(d) / total;
                toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
                if (Double.isNaN(toAdd)) {
                    toAdd = 0;
                }
                entropyParts.add(toAdd);
            }

            double entropy = 0;
            for (int i = 0; i < entropyParts.size(); i++) {
                entropy += entropyParts.get(i);
            }
            return entropy;
        }

        // comparison 1: to determine order of shapelets in terms of info gain, then separation gap, then shortness
        public int compareTo(Shapelet shapelet) {
            final int BEFORE = -1;
            final int EQUAL = 0;
            final int AFTER = 1;

            if (this.kruskalWallaceStat != shapelet.getKruskalWallisStat()) {
                if (this.kruskalWallaceStat > shapelet.getKruskalWallisStat()) {
                    return BEFORE;
                } else {
                    return AFTER;
                }
            } else if (this.content.length != shapelet.getLength()) {
                if (this.content.length < shapelet.getLength()) {
                    return BEFORE;
                } else {
                    return AFTER;
                }
            } else {
                return EQUAL;
            }
        }
    }

    private static class OrderLineObj implements Comparable<OrderLineObj> {

        private double distance;
        private double classVal;
        private double rank;

        private OrderLineObj(double distance, double classVal) {
            this.distance = distance;
            this.classVal = classVal;
            this.rank = -1;
        }

        public int compareTo(OrderLineObj o) {
            if (this.distance < o.distance) {
                return -1;
            } else if (this.distance == o.distance) {
                return 0;
            } else {
                return 1;
            }
        }

        public double getDistance() {
            return this.distance;
        }
    }

    private static void assignRanks(ArrayList<OrderLineObj> input) {

        double lastDistance = input.get(0).distance;
        double thisDistance = input.get(0).distance;
        input.get(0).rank = 1;

        int duplicateCount = 0;


        for (int i = 1; i < input.size(); i++) {
            thisDistance = input.get(i).distance;
            if (duplicateCount == 0 && thisDistance != lastDistance) { // standard entry
                input.get(i).rank = i + 1;
            } else if (duplicateCount > 0 && thisDistance != lastDistance) { // non-duplicate following duplicates
                // set ranks for dupicates

                double minRank = i - duplicateCount;
//                    System.out.println("minRank:"+minRank);
                double maxRank = i;
//                    System.out.println("maxRank:"+maxRank);
                double avgRank = (minRank + maxRank) / 2;

                for (int j = i - duplicateCount - 1; j < i; j++) {
                    input.get(j).rank = avgRank;
                }


                duplicateCount = 0;
                // then set this rank
                input.get(i).rank = i + 1;
            } else if (thisDistance == lastDistance) {
                if (i == input.size() - 1) { // last one so must do the avg ranks here (basically copied from above, BUT includes this element too now)
                    //                    System.out.println("NOT FINISHED"); // dw until normal conditions are confirmed working

                    double minRank = i - duplicateCount;
//                        System.out.println("minRank:"+minRank);
                    double maxRank = i + 1;
//                        System.out.println("maxRank:"+maxRank);
                    double avgRank = (minRank + maxRank) / 2;

                    for (int j = i - duplicateCount - 1; j <= i; j++) {
                        input.get(j).rank = avgRank;
                    }


                }
                duplicateCount++;
            } else {
                System.out.println("ERRORZ");
            }
            lastDistance = thisDistance;

        }
    }
}
