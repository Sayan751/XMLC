package Learner;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.SortedSet;
import java.util.stream.Collectors;

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

	private int treeDepth;

	// adaptable reference of the fields (lazy)
	private transient IAdaptiveHasher adaptiveFh;
	private transient IAdaptiveTuner adptTuner;
	private transient IAdaptiveTuner adptTestTopKTuner;
	private transient IAdaptiveTuner adptTestTuner;
	private transient AdaptiveTree adaptableTree;

	// followings are for temp variables; pulled out to avoid redefinition.
	// in adjustPropetiesWithGrowth:
	private List<Double> biasList;
	private List<Double> thresholdList;
	private List<Integer> tarrayList;
	private List<Double> scalararrayList;
	// in getTreeNodeIndexForLabel:
	private PriorityQueue<ComparablePair> positiveLabelsAndPosteriors;

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

		if (!(fh instanceof IAdaptiveHasher))
			throw new IllegalArgumentException(
					"Invalid init configuration: feature hasher must be of type IAdaptiveHasher.");

		if (!(thresholdTuner instanceof IAdaptiveTuner) || !(testTuner instanceof IAdaptiveTuner)
				|| !(testTopKTuner instanceof IAdaptiveTuner))
			throw new IllegalArgumentException("Threshold tuner is not of type IAdaptiveTuner");

		treeDepth = getAdaptableTree().getTreeDepth();
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
		int treeIndex = tree.getTreeIndex(label);
		if (treeIndex > -1)
			return treeIndex;

		int size = t;

		// choose a label and adapt
		positiveLabelsAndPosteriors = getPositiveLabelsAndPosteriors(instance.x);
		int newLeafIndex = getAdaptableTree().adaptLeaf(chooseLabel(positiveLabelsAndPosteriors, instance), label);

		t = getAdaptableTree().getSize();
		m = getAdaptableTree().getNumberOfLeaves();
		treeDepth = getAdaptableTree().getTreeDepth();

		adjustPropetiesWithGrowth(t - size);
		adjustTuner(label);

		logger.info("Tree structure adapted.");
		// logger.info(tree.toString());

		positiveLabelsAndPosteriors.clear();
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
	 * @param instance
	 * @return
	 */
	private int chooseLabel(PriorityQueue<ComparablePair> labelsAndPosteriors, Instance instance) {

		int retVal;

		if (!labelsAndPosteriors.isEmpty()) {
			retVal = chooseFromPredictedPositives(labelsAndPosteriors);
		} else {
			// choose any label from the instance.y that also exists in the tree
			Set<Integer> ys = new HashSet<Integer>(Ints.asList(instance.y));
			ys.retainAll(getAdaptableTree().getAllLabels());

			retVal = chooseFromTree(ys);
		}
		return retVal;
	}

	/**
	 * Chooses a leaf from the tree which is either the shallowest or deepest
	 * (controlled by {@code isToPreferShallowLeaf} )
	 * 
	 * @return
	 */
	private int chooseFromTree(Set<Integer> labels) {
		// double treeDepth = adaptableTree.getTreeDepth();
		return ((labels != null && !labels.isEmpty()) ? labels : getAdaptableTree().getAllLabels())
				.stream()
				.map(label -> {
					double relativeDepth = getAdaptableTree().getNodeDepth(getAdaptableTree().getTreeIndex(label))
							/ treeDepth;
					return new SimpleEntry<Integer, Double>(label,
							isToPreferShallowLeaf ? relativeDepth : 1 - relativeDepth);
				})
				.sorted(Entry.<Integer, Double>comparingByValue())
				.findFirst()
				.get()
				.getKey()
				.intValue();
	}

	/**
	 * Chooses label using the heuristic function:
	 * {@code alpha*prob + (1-alpha)*(1 - labelDepth/TreeDepth)}, when leaf with
	 * highest posterior is preferred, or
	 * {@code alpha*(1 - prob) + (1-alpha)*(1 - labelDepth/TreeDepth)}, when
	 * leaf with lowest posterior is preferred.
	 * 
	 * @param labelsAndPosteriors
	 * @return
	 */
	private int chooseFromPredictedPositives(PriorityQueue<ComparablePair> labelsAndPosteriors) {
		// double treeDepth = adaptableTree.getTreeDepth();
		return labelsAndPosteriors
				.stream()
				.map(n -> {
					double leafProb = n.getKey();
					int label = n.getValue();

					double labelDepth = getAdaptableTree().getNodeDepth(getAdaptableTree().getTreeIndex(label));

					double score = alpha * (isToPreferHighestProbLeaf ? leafProb : (1 - leafProb))
							+ (1 - alpha) * (1 - labelDepth / treeDepth);

					return new SimpleEntry<Integer, Double>(label, score);
				})
				.sorted(Entry.<Integer, Double>comparingByValue()
						.reversed())
				.collect(Collectors.toCollection(ArrayList<SimpleEntry<Integer, Double>>::new))
				.get(0)
				.getKey();
	}

	private void adjustTuner(int label) {
		getAdptTuner().accomodateNewLabel(label);
		getAdptTestTuner().accomodateNewLabel(label);
		getAdptTestTopKTuner().accomodateNewLabel(label);
	}

	private void adjustPropetiesWithGrowth(int growth) {
		if (growth > 0) {
			biasList = new ArrayList<>(bias.length);
			for (double item : bias)
				biasList.add(item);

			thresholdList = new ArrayList<>(thresholds.length);
			for (double item : thresholds)
				thresholdList.add(item);

			tarrayList = new ArrayList<>(Tarray.length);
			for (int item : Tarray)
				tarrayList.add(item);

			scalararrayList = new ArrayList<>(scalararray.length);
			for (double item : scalararray)
				scalararrayList.add(item);

			// biasList = Arrays.stream(bias)
			// .boxed()
			// .collect(Collectors.toList());
			// thresholdList = Arrays.stream(thresholds)
			// .boxed()
			// .collect(Collectors.toList());
			// tarrayList = Arrays.stream(Tarray)
			// .boxed()
			// .collect(Collectors.toList());
			// scalararrayList = Arrays.stream(scalararray)
			// .boxed()
			// .collect(Collectors.toList());
			for (int i = 0; i < growth; i++) {
				biasList.add(0.0);
				thresholdList.add(0.5);
				tarrayList.add(1);
				scalararrayList.add(1.0);
				getAdaptiveFh().adaptForNewTask();
			}
			// biases.addAll(Collections.nCopies(growth, 0.0));
			bias = Doubles.toArray(biasList);
			thresholds = Doubles.toArray(thresholdList);
			scalararray = Doubles.toArray(scalararrayList);
			Tarray = Ints.toArray(tarrayList);

			biasList.clear();
			thresholdList.clear();
			scalararrayList.clear();
			tarrayList.clear();
		}
	}

	private IAdaptiveHasher getAdaptiveFh() {
		if (adaptiveFh == null)
			adaptiveFh = (IAdaptiveHasher) fh;
		return adaptiveFh;
	}

	private IAdaptiveTuner getAdptTuner() {
		if (adptTuner == null)
			adptTuner = (IAdaptiveTuner) thresholdTuner;
		return adptTuner;
	}

	private IAdaptiveTuner getAdptTestTopKTuner() {
		if (adptTestTopKTuner == null)
			adptTestTopKTuner = (IAdaptiveTuner) testTopKTuner;
		return adptTestTopKTuner;
	}

	private IAdaptiveTuner getAdptTestTuner() {
		if (adptTestTuner == null)
			adptTestTuner = (IAdaptiveTuner) testTuner;
		return adptTestTuner;
	}

	private AdaptiveTree getAdaptableTree() {
		if (adaptableTree == null)
			adaptableTree = (AdaptiveTree) tree;
		return adaptableTree;
	}
}