package Learner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Ints;

import Data.AVPair;
//import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.Instance;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import IO.DataManager;
import preprocessing.FeatureHasher;
import preprocessing.FeatureHasherFactory;
import threshold.ThresholdTunerFactory;
import threshold.ThresholdTunerInitOption;
import threshold.ThresholdTuners;
import util.AdaptiveTree;
import util.CompleteTree;
import util.Constants.LearnerInitProperties;
import util.HuffmanTree;
import util.PLTInitConfiguration;
import util.PrecomputedTree;
import util.Tree;

public class PLT extends AbstractLearner {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(PLT.class);
	/**
	 * Number of node of the trees
	 */
	protected int t = 0;
	protected Tree tree = null;

	/**
	 * Order of the tree (k-ary).
	 */
	protected int k = 2;
	protected String treeType = "Complete";
	protected String treeFile = null;

	transient protected int T = 1;
	// transient protected AVTable traindata = null;
	transient protected DataManager traindata = null;

	protected FeatureHasher fh = null;
	protected String hasher = "Universal";
	protected int fhseed = 1;
	/**
	 * Hashed dimension, i.e. number of ML hashed features.
	 */
	protected int hd;

	protected double[] bias;
	protected double[] w = null;

	protected int[] Tarray = null;
	protected double[] scalararray = null;

	protected double gamma = 0; // learning rate
	transient protected int step = 0;
	static Sigmoid s = new Sigmoid();
	transient protected double learningRate = 1.0;
	protected double scalar = 1.0;
	protected double lambda = 0.00001;
	protected int epochs = 1;
	private ThresholdTuners tunerType;
	protected ThresholdTunerInitOption tunerInitOption;

	public PLT() {
		super();
	}

	public PLT(Properties properties) {
		super(properties);

		System.out.println("#####################################################");
		System.out.println("#### Learner: PLT");
		// learning rate
		this.gamma = Double.parseDouble(this.properties.getProperty("gamma", "1.0"));
		logger.info("#### gamma: " + this.gamma);

		// scalar
		this.lambda = Double.parseDouble(this.properties.getProperty("lambda", "1.0"));
		logger.info("#### lambda: " + this.lambda);

		// epochs
		this.epochs = Integer.parseInt(this.properties.getProperty("epochs", "30"));
		logger.info("#### epochs: " + this.epochs);

		// epochs
		this.hasher = this.properties.getProperty("hasher", "Mask");
		logger.info("#### Hasher: " + this.hasher);

		this.hd = Integer.parseInt(this.properties.getProperty("MLFeatureHashing", "50000000"));
		logger.info("#### Number of ML hashed features: " + this.hd);

		// k-ary tree
		this.k = Integer.parseInt(this.properties.getProperty("k", "2"));
		logger.info("#### k (order of the tree): " + this.k);

		// tree type (Complete, Precomputed, Huffman)
		this.treeType = this.properties.getProperty("treeType", "Complete");
		logger.info("#### tree type " + this.treeType);

		// tree file name
		this.treeFile = this.properties.getProperty("treeFile", null);
		logger.info("#### tree file name " + this.treeFile);

		tunerType = this.properties.containsKey(LearnerInitProperties.tunerType)
				? ThresholdTuners.valueOf(this.properties.getProperty(LearnerInitProperties.tunerType))
				: ThresholdTuners.None;

		tunerInitOption = (ThresholdTunerInitOption) properties
				.get(LearnerInitProperties.tunerInitOption);

		System.out.println("#####################################################");

	}

	public PLT(PLTInitConfiguration configuration) {
		super(configuration);

		// learning rate
		this.gamma = configuration.getGamma();

		// scalar
		this.lambda = configuration.getLambda();

		// epochs
		this.epochs = configuration.getEpochs();

		// epochs
		this.hasher = configuration.getHasher();

		this.hd = configuration.getHd();

		// k-ary tree
		this.k = configuration.getK();

		// tree type (Complete, Precomputed, Huffman)
		this.treeType = configuration.getTreeType();

		// tree file name
		this.treeFile = configuration.treeFile;

		tunerType = configuration.tunerType;
		tunerInitOption = configuration.tunerInitOption;
	}

	public void printParameters() {
		super.printParameters();
		logger.info("#### gamma: " + this.gamma);
		logger.info("#### lambda: " + this.lambda);
		logger.info("#### epochs: " + this.epochs);
		logger.info("#### Hasher: " + this.hasher);
		logger.info("#### Number of ML hashed features: " + this.hd);
		logger.info("#### k (order of the tree): " + this.k);
		logger.info("#### tree type: " + this.treeType);
		logger.info("#### tree file: " + this.treeFile);
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		allocateClassifiers(data, null);
	}

	public void allocateClassifiers(DataManager data, SortedSet<Integer> labels) {
		boolean labelsProvided = labels != null;
		this.traindata = data;
		if (labelsProvided)
			this.m = labels.size();
		else
			initializeNumberOfLabels(data);
		this.d = data.getNumberOfFeatures();

		this.tree = createTree(data, labels);
		this.t = this.tree.getSize();

		if (labelsProvided) {
			thresholdTuner = ThresholdTunerFactory.createThresholdTuner(labels, tunerType, tunerInitOption);
			testTopKTuner = ThresholdTunerFactory.createThresholdTuner(labels, tunerType, tunerInitOption);
			testTuner = ThresholdTunerFactory.createThresholdTuner(labels, tunerType, tunerInitOption);
		} else {
			thresholdTuner = ThresholdTunerFactory.createThresholdTuner(m, tunerType, tunerInitOption);
			testTopKTuner = ThresholdTunerFactory.createThresholdTuner(m, tunerType, tunerInitOption);
			testTuner = ThresholdTunerFactory.createThresholdTuner(m, tunerType, tunerInitOption);
		}

		logger.info("#### Num. of labels: " + this.m + " Dim: " + this.d);
		logger.info("#### Num. of node of the trees: " + this.t);
		logger.info("#####################################################");

		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);

		logger.info("Allocate the learners...");

		this.w = new double[this.hd];
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];

		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}

		if (thresholdTuner != null) {
			if (labelsProvided)
				thresholdTuner.getTunedThresholdsSparse(null)
						.forEach((label, threshold) -> {
							setThreshold(label, threshold);
						});
			else
				setThresholds(thresholdTuner.getTunedThresholds(null));
		}

		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
	}

	protected void initializeNumberOfLabels(DataManager data) {
		this.m = data.getNumberOfLabels();
	}

	protected Tree createTree(DataManager data) {
		return createTree(data, null);
	}

	protected Tree createTree(DataManager data, SortedSet<Integer> labels) {
		switch (this.treeType) {
		case CompleteTree.name:
			return labels == null || labels.isEmpty() ? new CompleteTree(this.k, this.m)
					: new AdaptiveTree(new CompleteTree(this.k, labels.size()), CompleteTree.name, labels);
		case PrecomputedTree.name:
			return new PrecomputedTree(this.treeFile);
		case HuffmanTree.name:
			return new HuffmanTree(data, this.treeFile);
		default:
			System.err.println("Unknown tree type!");
			System.exit(-1);
		}

		return null;
	}

	@Override
	public void train(DataManager data) {
		// prequential evaluation.
		evaluate(data, true);

		int oldT = T;
		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		// temp variables
		Instance instance;
		HashSet<Integer> positiveTreeIndices = new HashSet<Integer>(), negativeTreeIndices = new HashSet<Integer>();

		for (int ep = 0; ep < this.epochs; ep++) {

			// logger.info("#############--> BEGIN of Epoch: {} ({})", (ep + 1),
			// this.epochs);
			// random permutation

			while (data.hasNext() == true) {

				instance = data.getNextInstance();

				positiveTreeIndices.clear();
				negativeTreeIndices.clear();

				for (int j = 0; j < instance.y.length; j++) {
					// Labels start from 0
					int label = instance.y[j];
					/*this is kept outside if condition, to adapt the tree for unseen label (in case of AdaptivePLT)*/
					int treeIndex = getTreeNodeIndexForLabel(label, instance);
					if (tree.hasLabel(label)) {
						positiveTreeIndices.add(treeIndex);

						int rootIndex = tree.getRootIndex();
						while (treeIndex != rootIndex) {
							treeIndex = (int) this.tree.getParent(treeIndex);
							positiveTreeIndices.add(treeIndex);
						}
					}
				}

				if (positiveTreeIndices.size() == 0) {
					negativeTreeIndices.add(this.tree.getRootIndex());
				} else {
					for (int positiveNode : positiveTreeIndices) {
						if (!this.tree.isLeaf(positiveNode)) {
							for (int childNode : this.tree.getChildNodes(positiveNode)) {
								if (!positiveTreeIndices.contains(childNode)) {
									negativeTreeIndices.add(childNode);
								}
							}
						}
					}
				}

				for (int j : positiveTreeIndices) {

					double posterior = getPartialPosteriors(instance.x, j);
					double inc = -(1.0 - posterior);

					updatedPosteriors(instance.x, j, inc);
				}

				for (int j : negativeTreeIndices) {

					if (j >= this.t)
						logger.info("ALARM");

					double posterior = getPartialPosteriors(instance.x, j);
					double inc = -(0.0 - posterior);

					updatedPosteriors(instance.x, j, inc);
				}

				this.T++;

				if ((this.T % 100000) == 0) {
					logger.info("\t --> Epoch: " + (ep + 1) + " (" + this.epochs + ")" + "\tSample: " + this.T);
				}
			}
			if (ep == 0) {
				nTrain += (this.T - oldT);
			}

			data.reset();

			// logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs
			// + ")");
		}
		if (measureTime) {
			getStopwatch().stop();
			totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}

		int zeroW = 0;
		double sumW = 0;
		int maxNonZero = 0;
		int index = 0;
		for (double weight : w) {
			if (weight == 0)
				zeroW++;
			else
				maxNonZero = index;
			sumW += weight;
			index++;
		}
		logger.info("Hash weights (lenght, zeros, nonzeros, ratio, sumW, last nonzero): " + w.length + ", " + zeroW
				+ ", " + (w.length - zeroW) + ", " + (double) (w.length - zeroW) / (double) w.length + ", " + sumW
				+ ", " + maxNonZero);

		// tuning thresholds from learner is optional as of now. if made
		// mandatory, then this kind of checks can be removed.
		if (this.thresholdTuner != null) {
			if (measureTime) {
				getStopwatch().reset();
				getStopwatch().start();
			}

			tuneThreshold(data);

			if (measureTime) {
				getStopwatch().stop();
				totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
			}
		}

		evaluate(data, false);
	}

	public void train(Instance instance) {
		train(instance, this.epochs, true);
	}

	public void train(Instance instance, int epochs, boolean toEvaluate) {

		// prequential evaluation.
		if (toEvaluate)
			evaluate(instance, true);

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		HashSet<Integer> positiveTreeIndices = new HashSet<Integer>(), negativeTreeIndices = new HashSet<Integer>();

		for (int ep = 0; ep < epochs; ep++) {

			positiveTreeIndices.clear();
			negativeTreeIndices.clear();

			for (int j = 0; j < instance.y.length; j++) {
				// Labels start from 0
				int label = instance.y[j];
				/*this is kept outside if condition, to adapt the tree for unseen label (in case of AdaptivePLT)*/
				int treeIndex = getTreeNodeIndexForLabel(label, instance);
				if (this.tree.hasLabel(label)) {
					positiveTreeIndices.add(treeIndex);

					int rootIndex = tree.getRootIndex();
					while (treeIndex != rootIndex) {
						treeIndex = (int) this.tree.getParent(treeIndex);
						positiveTreeIndices.add(treeIndex);
					}
				}
			}

			if (positiveTreeIndices.size() == 0) {
				negativeTreeIndices.add(this.tree.getRootIndex());
			} else {
				for (int positiveNode : positiveTreeIndices) {
					if (!this.tree.isLeaf(positiveNode)) {
						for (int childNode : this.tree.getChildNodes(positiveNode)) {
							if (!positiveTreeIndices.contains(childNode)) {
								negativeTreeIndices.add(childNode);
							}
						}
					}
				}
			}

			for (int j : positiveTreeIndices) {

				double posterior = getPartialPosteriors(instance.x, j);
				double inc = -(1.0 - posterior);

				updatedPosteriors(instance.x, j, inc);
			}

			for (int j : negativeTreeIndices) {

				if (j >= this.t)
					logger.info("ALARM");

				double posterior = getPartialPosteriors(instance.x, j);
				double inc = -(0.0 - posterior);

				updatedPosteriors(instance.x, j, inc);
			}
		}
		if (measureTime) {
			getStopwatch().stop();
			totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}
		nTrain++;

		// int zeroW = 0;
		// double sumW = 0;
		// int maxNonZero = 0;
		// int index = 0;
		// for (double weight : w) {
		// if (weight == 0)
		// zeroW++;
		// else
		// maxNonZero = index;
		// sumW += weight;
		// index++;
		// }
		// logger.info("Hash weights (lenght, zeros, nonzeros, ratio, sumW, last
		// nonzero): " + w.length + ", " + zeroW
		// + ", " + (w.length - zeroW) + ", " + (double) (w.length - zeroW) /
		// (double) w.length + ", " + sumW
		// + ", " + maxNonZero);

		// tuning thresholds from learner is optional as of now. if made
		// mandatory, then this kind of checks can be removed.
		if (this.thresholdTuner != null) {
			if (measureTime) {
				getStopwatch().reset();
				getStopwatch().start();
			}
			tuneThreshold(instance);
			if (measureTime) {
				getStopwatch().stop();
				totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
			}
		}

		if (toEvaluate)
			evaluate(instance, false);
	}

	protected Integer getTreeNodeIndexForLabel(int label, Instance instance) {
		try {
			return this.tree.getTreeIndex(label);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	protected void updatedPosteriors(AVPair[] x, int label, double inc) {

		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);

		int n = x.length;

		for (int i = 0; i < n; i++) {

			int index = fh.getIndex(label, x[i].index);
			int sign = fh.getSign(label, x[i].index);

			double gradient = this.scalararray[label] * inc * (x[i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;
			this.w[index] -= update;

		}

		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);// / this.scalar;
		this.bias[label] -= update;
		// logger.info("bias -> gradient, scalar, update: " + gradient + ", " +
		// scalar +", " + update);
	}

	/**
	 * Computes and returns {@code sigmoid(fh(x).dot(weight) + bias[label])}
	 * 
	 * @param x
	 *            Sparse feature vector.
	 * @param label
	 *            Node index of PLT.
	 * @return Class (1 or 0) probability estimate at node {@code label} for the
	 *         given instance {@code x}, as per current weight vector and bias
	 *         at node {@code label}.
	 */
	public double getPartialPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;

		for (int i = 0; i < x.length; i++) {

			int hi = fh.getIndex(label, x[i].index);
			int sign = fh.getSign(label, x[i].index);
			posterior += (x[i].value * sign) * (1 / this.scalararray[label]) * this.w[hi];
		}

		posterior += (1 / this.scalararray[label]) * this.bias[label];
		posterior = s.value(posterior);

		return posterior;

	}

	protected Object readResolve() {
		switch (this.treeType) {
		case CompleteTree.name:
			this.tree = new CompleteTree(this.k, this.m);
			break;
		case PrecomputedTree.name:
			this.tree = new PrecomputedTree(this.treeFile);
			break;
		case HuffmanTree.name:
			this.tree = new HuffmanTree(this.treeFile);
			break;
		default:
			System.err.println("Unknown tree type!");
			System.exit(-1);
		}
		this.t = this.tree.getSize();
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);
		return this;
	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 1.0;

		int treeIndex = this.tree.getTreeIndex(label);

		posterior *= getPartialPosteriors(x, treeIndex);

		int rootIndex = tree.getRootIndex();
		while (treeIndex != rootIndex) {

			treeIndex = this.tree.getParent(treeIndex); // Math.floor((treeIndex
														// - 1)/2);
			posterior *= getPartialPosteriors(x, treeIndex);

		}
		// if(posterior > 0.5) logger.info("Posterior: " + posterior + "Label: "
		// + label);
		return posterior;
	}

	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(tree.getRootIndex(), 1.0));

		while (!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if (currentP >= this.thresholds[node.treeIndex]) {

				if (!this.tree.isLeaf(node.treeIndex)) {

					for (int childNode : this.tree.getChildNodes(node.treeIndex)) {
						queue.add(new NodePLT(childNode, currentP));
					}

				} else {

					int labelIndex = this.tree.getLabelIndex(node.treeIndex);
					/*Assumption: label index must be greater than or equal to 0.*/
					if (labelIndex > -1)
						positiveLabels.add(labelIndex);

				}
			}
		}

		// logger.info("Predicted labels: " + positiveLabels);

		return positiveLabels;
	}

	@Override
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();

		NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(tree.getRootIndex(), 1.0));

		while (!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if (currentP > this.thresholds[node.treeIndex]) {

				if (!this.tree.isLeaf(node.treeIndex)) {

					for (int childNode : this.tree.getChildNodes(node.treeIndex)) {
						queue.add(new NodePLT(childNode, currentP));
					}

				} else {

					int labelIndex = this.tree.getLabelIndex(node.treeIndex);
					/*Assumption: label index must be greater than or equal to 0.*/
					if (labelIndex > -1)
						positiveLabels.add(new ComparablePair(currentP, labelIndex));

				}
			}
		}

		// logger.info("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}

	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		/*this avoids having k number of 0s in output in case there are no valid predicted positives.*/
		List<Integer> positiveLabels = new ArrayList<>();

		NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<>(11, nodeComparator);

		queue.add(new NodePLT(tree.getRootIndex(), 1.0));

		while (!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if (!this.tree.isLeaf(node.treeIndex)) {

				for (int childNode : this.tree.getChildNodes(node.treeIndex)) {
					queue.add(new NodePLT(childNode, currentP));
				}

			} else {
				int labelIndex = this.tree.getLabelIndex(node.treeIndex);
				/*Assumption: label index must be greater than or equal to 0.*/
				if (labelIndex > -1)
					positiveLabels.add(labelIndex);
			}

			if (positiveLabels.size() >= k) {
				break;
			}
		}

		// int[] positiveLabelsArray = Ints.toArray(positiveLabels);

		// logger.info("Predicted labels: " +
		// Arrays.toString(positiveLabelsArray));

		// return positiveLabelsArray;
		return Ints.toArray(positiveLabels);
	}

	@Override
	public void setThreshold(int label, double t) {

		int treeIndex = this.tree.getTreeIndex(label);
		this.thresholds[treeIndex] = t;

		while (!tree.isRoot(treeIndex)) {
			treeIndex = this.tree.getParent(treeIndex);

			double minThreshold = Double.MAX_VALUE;
			for (int childNode : this.tree.getChildNodes(treeIndex)) {
				minThreshold = this.thresholds[childNode] < minThreshold ? this.thresholds[childNode]
						: minThreshold;
			}
			this.thresholds[treeIndex] = minThreshold;

		}

	}

	public void setThresholds(double[] t) {

		for (int j = 0; j < t.length; j++) {
			this.thresholds[this.tree.getTreeIndex(j)] = t[j];
		}

		if (tree instanceof AdaptiveTree)
			getSetThreshold(tree.getRootIndex());
		else
			for (int j = this.tree.getNumberOfInternalNodes() - 1; j >= 0; j--) {

				double minThreshold = Double.MAX_VALUE;
				for (int childNode : this.tree.getChildNodes(j)) {
					minThreshold = this.thresholds[childNode] < minThreshold ? this.thresholds[childNode]
							: minThreshold;
				}

				this.thresholds[j] = minThreshold;
			}

		// for( int i=0; i < this.thresholds.length; i++ )
		// logger.info( "Threshold: " + i + " Th: " + String.format("%.4f",
		// this.thresholds[i]) );

	}

	private double getSetThreshold(int nodeIndex) {
		if (tree.isLeaf(nodeIndex))
			return thresholds[nodeIndex];
		else {
			double minThreshold = tree.getChildNodes(nodeIndex)
					.stream()
					.mapToDouble(childNodeIndex -> getSetThreshold(childNodeIndex))
					.min()
					.getAsDouble();
			thresholds[nodeIndex] = minThreshold;
			return minThreshold;
		}
	}

	@Override
	public HashSet<EstimatePair> getSparseProbabilityEstimates(AVPair[] x, double threshold) {

		HashSet<EstimatePair> positiveLabels = new HashSet<EstimatePair>();

		NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(tree.getRootIndex(), 1.0));

		while (!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if (currentP >= threshold) {

				if (!this.tree.isLeaf(node.treeIndex)) {

					for (int childNode : this.tree.getChildNodes(node.treeIndex)) {
						queue.add(new NodePLT(childNode, currentP));
					}

				} else {

					int labelIndex = this.tree.getLabelIndex(node.treeIndex);
					/*Assumption: label index must be greater than or equal to 0.*/
					if (labelIndex > -1)
						positiveLabels.add(new EstimatePair(labelIndex, currentP));

				}
			}
		}

		// logger.info("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}

	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {

		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();

		int foundTop = 0;

		NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(tree.getRootIndex(), 1.0));

		while (!queue.isEmpty() && (foundTop < k)) {

			NodePLT node = queue.poll();

			double currentP = node.p;

			if (!this.tree.isLeaf(node.treeIndex)) {

				for (int childNode : this.tree.getChildNodes(node.treeIndex)) {
					queue.add(new NodePLT(childNode, currentP * getPartialPosteriors(x, childNode)));
				}

			} else {

				int labelIndex = this.tree.getLabelIndex(node.treeIndex);
				/*Assumption: label index must be greater than or equal to 0.*/
				if (labelIndex > -1) {
					positiveLabels.add(new EstimatePair(labelIndex, currentP));
					foundTop++;
				}

			}
		}

		// logger.info("Top k positive labels: " + positiveLabels);
		return positiveLabels;
	}

	/**
	 * Provides top k labels and estimates.
	 * 
	 * <br/>
	 * <b>Note:</b> This new method can enables putting labels with exactly same
	 * posterior estimates in the final output set, unlike
	 * {@link PLT#getTopKEstimates(AVPair[], int)}, where this is not possible.
	 * 
	 * @param x
	 *            Feature vector.
	 * @param k
	 *            Size of predicion set.
	 */
	public List<EstimatePair> getTopKEstimatesComplete(AVPair[] x, int k) {

		Set<EstimatePair> positiveLabels = new HashSet<EstimatePair>();

		// int foundTop = 0;

		NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(tree.getRootIndex(), 1.0));

		while (!queue.isEmpty()) {// && (foundTop < k)

			NodePLT node = queue.poll();

			double currentP = node.p;

			if (!this.tree.isLeaf(node.treeIndex)) {

				for (int childNode : this.tree.getChildNodes(node.treeIndex)) {
					queue.add(new NodePLT(childNode, currentP * getPartialPosteriors(x, childNode)));
				}

			} else {

				int labelIndex = this.tree.getLabelIndex(node.treeIndex);
				/*Assumption: label index must be greater than or equal to 0.*/
				if (labelIndex > -1) {
					positiveLabels.add(new EstimatePair(labelIndex, currentP));
					// foundTop++;
				}

			}
		}

		// logger.info("Top k positive labels: " + positiveLabels);
		return positiveLabels.stream()
				.sorted()
				.collect(Collectors.toList());
	}
}
