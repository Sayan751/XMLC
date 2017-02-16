package Learner;

import java.util.AbstractMap.SimpleEntry;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Doubles;

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

	public AdaptivePLT() {
	}

	public AdaptivePLT(Properties properties) {
		super(properties);

		isToPreferHighestProbLeaf = Boolean.parseBoolean(properties.getProperty(
				LearnerInitProperties.isToPreferHighestProbLeaf, AdaptivePLTDefaultValues.isToPreferHighestProbLeaf));
		probabilityWeight = Double.parseDouble(properties.getProperty(
				LearnerInitProperties.probabilityWeight, AdaptivePLTDefaultValues.probabilityWeight));
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
	protected int getTreeNodeIndexForLabel(int label, Instance instance) {
		if (label < m)
			return super.getTreeNodeIndexForLabel(label, instance);

		int size = tree.getSize();

		//choose a label and adapt
		int chosenLabelIndex = chooseLabel(getPositiveLabelsAndPosteriors(instance.x), this.tree);
		int newLeafIndex = ((AdaptableTree) this.tree).adaptLeaf(chosenLabelIndex, label);

		int growth = tree.getSize() - size;

		//adjust bias
		List<Double> biases = Doubles.asList(bias);
		biases.addAll(Collections.nCopies(growth, 0.0));
		bias = Doubles.toArray(biases);

		//adjust tuner
		IAdaptiveTuner tuner = (IAdaptiveTuner) this.thresholdTuner;
		if (tuner != null) {
			tuner.accomodateNewLabel(label);
		} else {
			logger.error("Threshold tuner is not of type IAdaptiveTuner");
			System.exit(-1);
		}

		return newLeafIndex;
	}

	private int chooseLabel(PriorityQueue<ComparablePair> labelsAndPosteriors, Tree tree) {

		AdaptableTree adpatableTree = ((AdaptableTree) tree);
		double treeDepth = adpatableTree.getTreeDepth();

		return labelsAndPosteriors
				.parallelStream()
				.map(n -> {
					double leafProb = n.getKey();
					int labelIndex = n.getValue();

					double labelDepth = adpatableTree.getNodeDepth(adpatableTree.getTreeIndex(labelIndex));

					double score = probabilityWeight * (isToPreferHighestProbLeaf ? leafProb : (1 - leafProb))
							+ (1 - probabilityWeight) * (1 - labelDepth / treeDepth);

					return new SimpleEntry<Integer, Double>(labelIndex, score);
				})
				.sorted(Entry.<Integer, Double>comparingByValue()
						.reversed())
				.findFirst()
				.get()
				.getKey();
	}
}
