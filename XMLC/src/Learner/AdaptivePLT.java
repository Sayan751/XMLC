package Learner;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;

import Data.ComparablePair;
import Data.Instance;
import IO.DataManager;
import threshold.IAdaptiveTuner;
import util.AdaptableTree;
import util.Constants.AdaptivePLTDefaultValues;
import util.Constants.LearnerInitProperties;
import util.Tree;

public class AdaptivePLT extends PLT {
	private static final long serialVersionUID = 7227495342317859325L;
	private static Logger logger = LoggerFactory.getLogger(PLT.class);

	private boolean isToPreferHighestProbLeaf;
	/**
	 * Weight to put on probability of leaf, while choosing a leaf to adapt
	 * (alpha).
	 */
	private double probabilityWeight;
	private boolean isToPreferShallowLeaf;
	private Random random = new Random();

	public AdaptivePLT() {
	}

	public AdaptivePLT(Properties properties) {
		super(properties);

		isToPreferHighestProbLeaf = Boolean.parseBoolean(properties.getProperty(
				LearnerInitProperties.isToPreferHighestProbLeaf, AdaptivePLTDefaultValues.isToPreferHighestProbLeaf));
		probabilityWeight = Double.parseDouble(properties.getProperty(
				LearnerInitProperties.probabilityWeight, AdaptivePLTDefaultValues.probabilityWeight));
		isToPreferShallowLeaf = Boolean.parseBoolean(properties.getProperty(
				LearnerInitProperties.isToPreferShallowLeaf, AdaptivePLTDefaultValues.isToPreferShallowLeaf));
	}

	@Override
	protected Tree createTree(DataManager data) {
		try {
			return new AdaptableTree(super.createTree(data), treeType);
		} catch (Exception e) {
			logger.error("Error in creating tree", e);
		}
		return null;
	}

	@Override
	protected Integer getTreeNodeIndexForLabel(int label, Instance instance) {
		int treeIndex = this.tree.getTreeIndex(label);
		if (treeIndex > -1)
			return treeIndex;

		int size = t;

		// choose a label and adapt
		PriorityQueue<ComparablePair> positiveLabelsAndPosteriors = getPositiveLabelsAndPosteriors(instance.x);
		int chosenLabelIndex = chooseLabel(positiveLabelsAndPosteriors, this.tree, instance);
		int newLeafIndex = ((AdaptableTree) this.tree).adaptLeaf(chosenLabelIndex, label);

		t = tree.getSize();
		m = this.tree.getNumberOfLeaves();

		adjustPropetiesWithGrowth(t - size);
		adjustTuner(label);

		logger.info("Tree structure adapted.");
		logger.info(tree.toString());

		return newLeafIndex;
	}

	/**
	 * Chooses a label to adapt. The label is chosen in this order: <br/>
	 * <ol>
	 * <li>At first {@code labelsAndPosteriors} is searched for a suitable
	 * label.</li>
	 * <li>If {@code labelsAndPosteriors} is empty, then {@code instance.y}
	 * (instance's label set) is searched for any random label, which also
	 * exists in the tree.</li>
	 * <li>If no such label exists, then either the shallowest or the deepest
	 * leaf/label (controlled by {@code isToPreferShallowLeaf}) is chosen from
	 * the tree.</li>
	 * </ol>
	 * 
	 * @param labelsAndPosteriors
	 * @param tree
	 * @param instance
	 * @return
	 */
	private int chooseLabel(PriorityQueue<ComparablePair> labelsAndPosteriors, Tree tree, Instance instance) {

		AdaptableTree adaptableTree = ((AdaptableTree) tree);
		int retVal;

		if (!labelsAndPosteriors.isEmpty()) {
			retVal = chooseFromPredictedPositives(labelsAndPosteriors, adaptableTree);
		} else {
			// choose any label from the instance.y that also exists in the tree
			Set<Integer> ys = IntStream.of(instance.y)
					.boxed()
					.collect(Collectors.toCollection(HashSet<Integer>::new));
			ys.retainAll(adaptableTree.getAllLabels());

			if (!ys.isEmpty()) {
				retVal = ys.stream()
						.findAny()
						.get()
						.intValue();
			} else {
				/*else pickup a leaf from the tree  which is either the shallowest or deepest.*/
				retVal = chooseFromTree(adaptableTree);
			}
		}
		return retVal;
	}

	/**
	 * Chooses a leaf from the tree which is either the shallowest or deepest
	 * (controlled by {@code isToPreferShallowLeaf} ).
	 * 
	 * @param adaptableTree
	 * @return
	 */
	private int chooseFromTree(AdaptableTree adaptableTree) {
		int retVal;
		double treeDepth = adaptableTree.getTreeDepth();
		retVal = adaptableTree.getAllLabels()
				.stream()
				.map(label -> {
					double relativeDepth = adaptableTree.getNodeDepth(adaptableTree.getTreeIndex(label))
							/ treeDepth;
					return new SimpleEntry<Integer, Double>(label,
							isToPreferShallowLeaf ? relativeDepth : 1 - relativeDepth);
				})
				.sorted(Entry.<Integer, Double>comparingByValue())
				.findFirst()
				.get()
				.getKey()
				.intValue();
		return retVal;
	}

	/**
	 * Chooses label using the heuristic function:
	 * {@code alpha*prob + (1-alpha)*(1 - labelDepth/TreeDepth)}, when leaf with
	 * highest posterior is preferred, or
	 * {@code alpha*(1 - prob) + (1-alpha)*(1 - labelDepth/TreeDepth)}, when
	 * leaf with lowest posterior is preferred.
	 * 
	 * @param labelsAndPosteriors
	 * @param adaptableTree
	 * @return
	 */
	private int chooseFromPredictedPositives(PriorityQueue<ComparablePair> labelsAndPosteriors,
			AdaptableTree adaptableTree) {
		int retVal;
		double treeDepth = adaptableTree.getTreeDepth();

		Stream<SimpleEntry<Integer, Double>> sorted = labelsAndPosteriors
				.stream()
				.map(n -> {
					double leafProb = n.getKey();
					int labelIndex = n.getValue();

					double labelDepth = adaptableTree.getNodeDepth(adaptableTree.getTreeIndex(labelIndex));

					double score = probabilityWeight * (isToPreferHighestProbLeaf ? leafProb : (1 - leafProb))
							+ (1 - probabilityWeight) * (1 - labelDepth / treeDepth);

					return new SimpleEntry<Integer, Double>(labelIndex, score);
				})
				.sorted(Entry.<Integer, Double>comparingByValue()
						.reversed());
		ArrayList<SimpleEntry<Integer, Double>> test = sorted
				.collect(Collectors.toCollection(ArrayList<SimpleEntry<Integer, Double>>::new));
		retVal = test.get(0)
				.getKey();
		return retVal;
	}

	private void adjustTuner(int label) {
		// adjust tuner
		IAdaptiveTuner tuner = (IAdaptiveTuner) this.thresholdTuner;
		if (tuner != null) {
			tuner.accomodateNewLabel(label);
		} else {
			logger.error("Threshold tuner is not of type IAdaptiveTuner");
			System.exit(-1);
		}
	}

	private void adjustPropetiesWithGrowth(int growth) {
		// adjust bias
		List<Double> biasList = Arrays.stream(bias)
				.boxed()
				.collect(Collectors.toList());
		List<Double> thresholdList = Arrays.stream(thresholds)
				.boxed()
				.collect(Collectors.toList());
		List<Integer> TarrayList = Arrays.stream(Tarray)
				.boxed()
				.collect(Collectors.toList());
		List<Double> scalararrayList = Arrays.stream(scalararray)
				.boxed()
				.collect(Collectors.toList());
		for (int i = 0; i < growth; i++) {
			biasList.add(0.0);
			thresholdList.add(0.5);
			TarrayList.add(1);
			scalararrayList.add(1.0);
		}
		// biases.addAll(Collections.nCopies(growth, 0.0));
		bias = Doubles.toArray(biasList);
		thresholds = Doubles.toArray(thresholdList);
		scalararray = Doubles.toArray(scalararrayList);
		Tarray = Ints.toArray(TarrayList);
	}
}
