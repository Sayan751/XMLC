package Learner;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.SortedSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;

import Data.ComparablePair;
import Data.Instance;
import IO.DataManager;
import preprocessing.IAdaptiveHasher;
import threshold.IAdaptiveTuner;
import util.AdaptivePLTInitConfiguration;
import util.AdaptiveTree;
import util.Tree;

public class AdaptivePLT extends PLT {
	private static final long serialVersionUID = 7227495342317859325L;
	private static Logger logger = LoggerFactory.getLogger(PLT.class);

	/**
	 * Whether to prefer the node with the highest posterior or the lowest
	 * posterior while adapting the PLT.
	 */
	private boolean isToPreferHighestProbLeaf;
	/**
	 * Weight to put on probability of leaf, while choosing a leaf to adapt.
	 */
	private double alpha;
	/**
	 * Whether to prefer the shallowest node or the deepest node in the PLT.
	 */
	private boolean isToPreferShallowLeaf;

	public AdaptivePLT() {
	}

	public AdaptivePLT(AdaptivePLTInitConfiguration configuration) {
		super(configuration);

		isToPreferHighestProbLeaf = configuration.isToPreferHighestProbLeaf();
		alpha = configuration.getAlpha();
		isToPreferShallowLeaf = configuration.isToPreferShallowLeaf();
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		super.allocateClassifiers(data);
		if ((IAdaptiveHasher) fh == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: feature hasher must be of type IAdaptiveHasher.");
	}

	@Override
	protected void initializeNumberOfLabels(DataManager data) {
		/*Start from scratch with no label.*/
		this.m = 0;
	}

	@Override
	protected Tree createTree(DataManager data, SortedSet<Integer> labels) {
		Tree tr = super.createTree(data, labels);
		return tr instanceof AdaptiveTree ? tr : new AdaptiveTree(tr, treeType, shuffleLabels);
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
		int newLeafIndex = ((AdaptiveTree) this.tree).adaptLeaf(chosenLabelIndex, label);

		t = tree.getSize();
		m = this.tree.getNumberOfLeaves();

		adjustPropetiesWithGrowth(t - size);
		adjustTuner(label);

		logger.info("Tree structure adapted.");
		//logger.info(tree.toString());

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

		AdaptiveTree adaptableTree = ((AdaptiveTree) tree);
		int retVal;

		if (!labelsAndPosteriors.isEmpty()) {
			retVal = chooseFromPredictedPositives(labelsAndPosteriors, adaptableTree);
		} else {
			// choose any label from the instance.y that also exists in the tree
			Set<Integer> ys = new HashSet<Integer>(Ints.asList(instance.y));
			ys.retainAll(adaptableTree.getAllLabels());

			retVal = chooseFromTree(adaptableTree, ys);
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
	private int chooseFromTree(AdaptiveTree adaptableTree, Set<Integer> labels) {
		int retVal;
		double treeDepth = adaptableTree.getTreeDepth();
		Set<Integer> labelSet = (labels != null && !labels.isEmpty()) ? labels : adaptableTree.getAllLabels();
		retVal = labelSet
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
			AdaptiveTree adaptableTree) {
		int retVal;
		double treeDepth = adaptableTree.getTreeDepth();

		Stream<SimpleEntry<Integer, Double>> sorted = labelsAndPosteriors
				.stream()
				.map(n -> {
					double leafProb = n.getKey();
					int label = n.getValue();

					double labelDepth = adaptableTree.getNodeDepth(adaptableTree.getTreeIndex(label));

					double score = alpha * (isToPreferHighestProbLeaf ? leafProb : (1 - leafProb))
							+ (1 - alpha) * (1 - labelDepth / treeDepth);

					return new SimpleEntry<Integer, Double>(label, score);
				})
				.sorted(Entry.<Integer, Double>comparingByValue()
						.reversed());
		retVal = sorted
				.collect(Collectors.toCollection(ArrayList<SimpleEntry<Integer, Double>>::new))
				.get(0)
				.getKey();
		return retVal;
	}

	private void adjustTuner(int label) {
		// adjust tuner
		IAdaptiveTuner tuner = thresholdTuner instanceof IAdaptiveTuner ? (IAdaptiveTuner) thresholdTuner : null;
		if (tuner != null)
			tuner.accomodateNewLabel(label);
		else
			throw new IllegalArgumentException("Threshold tuner is not of type IAdaptiveTuner");
	}

	private void adjustPropetiesWithGrowth(int growth) {
		if (growth > 0) {
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
			IAdaptiveHasher adaptiveFh = (IAdaptiveHasher) fh;
			for (int i = 0; i < growth; i++) {
				biasList.add(0.0);
				thresholdList.add(0.5);
				TarrayList.add(1);
				scalararrayList.add(1.0);
				adaptiveFh.adaptForNewTask();
			}
			// biases.addAll(Collections.nCopies(growth, 0.0));
			bias = Doubles.toArray(biasList);
			thresholds = Doubles.toArray(thresholdList);
			scalararray = Doubles.toArray(scalararrayList);
			Tarray = Ints.toArray(TarrayList);
		}
	}
}