package threshold;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import util.Constants.OFODefaultValues;
import util.Constants.ThresholdTuningDataKeys;

/**
 * Tunes thresholds using online F-Measure optimization algorithm. This is a
 * 'flexible' version of {@link OfoFastThresholdTuner}, as it uses Map instead
 * of arrays, for the tuning thresholds.
 * 
 * @author Sayan
 *
 */
public class AdaptiveOfoFastThresholdTuner extends ThresholdTuner implements IAdaptiveTuner {
	private static Logger logger = LoggerFactory.getLogger(AdaptiveOfoFastThresholdTuner.class);

	/**
	 * Label to numerator mapping.
	 */
	protected Map<Integer, Integer> aThresholdNumerators = null;
	/**
	 * Label to denominator mapping.
	 */
	protected Map<Integer, Integer> bThresholdDenominators = null;
	int aSeed, bSeed;

	public AdaptiveOfoFastThresholdTuner() {
	}

	public AdaptiveOfoFastThresholdTuner(SortedSet<Integer> labels, ThresholdTunerInitOption thresholdTunerInitOption) {
		super(labels.size(), thresholdTunerInitOption);
		init(labels.stream(), thresholdTunerInitOption);
	}

	public AdaptiveOfoFastThresholdTuner(int numberOfLabels, ThresholdTunerInitOption thresholdTunerInitOption) {
		super(numberOfLabels, thresholdTunerInitOption);
		init(IntStream.range(0, this.numberOfLabels)
				.boxed(), thresholdTunerInitOption);
	}

	private void init(Stream<Integer> labelStream,
			ThresholdTunerInitOption thresholdTunerInitOption) {

		logger.info("#####################################################");
		logger.info("#### OFO Fast");
		logger.info("#### numberOfLabels: " + this.numberOfLabels);

		aSeed = thresholdTunerInitOption != null && thresholdTunerInitOption.aSeed != null
				? thresholdTunerInitOption.aSeed : OFODefaultValues.aSeed;

		bSeed = thresholdTunerInitOption != null && thresholdTunerInitOption.bSeed != null
				? thresholdTunerInitOption.bSeed : OFODefaultValues.bSeed;

		aThresholdNumerators = new HashMap<Integer, Integer>();
		bThresholdDenominators = new HashMap<Integer, Integer>();

		int[] aInit = thresholdTunerInitOption.aInit;
		int[] bInit = thresholdTunerInitOption.bInit;

		if (thresholdTunerInitOption != null
				&& aInit != null && aInit.length > 0 && aInit.length == this.numberOfLabels
				&& bInit != null && bInit.length > 0 && bInit.length == this.numberOfLabels) {

			labelStream.forEach(label -> accomodateNewLabel(label, aInit[label], bInit[label]));

			logger.info("#### a[] and b[] are initialized with predefined values");

		} else {

			labelStream.forEach(label -> accomodateNewLabel(label));

			logger.info("#### a[] seed: " + aSeed);
			logger.info("#### b[] seed: " + bSeed);
		}

		logger.info("#####################################################");
	}

	@Override
	public ThresholdTuners getTunerType() {
		return ThresholdTuners.AdaptiveOfoFast;
	}

	@Override
	public void accomodateNewLabel(int label) {
		accomodateNewLabel(label, aSeed, bSeed);
	}

	@Override
	public void accomodateNewLabel(int label, int a, int b) {
		aThresholdNumerators.put(label, a);
		bThresholdDenominators.put(label, b);
	}

	@Override
	public double[] getTunedThresholds(Map<String, Object> tuningData) {

		if (tuningData != null) {
			@SuppressWarnings("unchecked")
			List<HashSet<Integer>> predictedLabels = (List<HashSet<Integer>>) tuningData
					.get(ThresholdTuningDataKeys.predictedLabels);

			@SuppressWarnings("unchecked")
			List<HashSet<Integer>> trueLabels = (List<HashSet<Integer>>) tuningData
					.get(ThresholdTuningDataKeys.trueLabels);

			if (predictedLabels != null || trueLabels != null) {

				tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			}
		}

		double[] thresholds = new double[aThresholdNumerators.size()];

		for (int label = 0; label < aThresholdNumerators.size(); label++) {
			thresholds[label] = (double) aThresholdNumerators.get(label) / (double) bThresholdDenominators.get(label);
		}

		return thresholds;
	}

	@Override
	public Map<Integer, Double> getTunedThresholdsSparse(Map<String, Object> tuningData) throws Exception {
		Set<Integer> thresholdsToChange = null;
		if (tuningData != null) {

			@SuppressWarnings("unchecked")
			List<HashSet<Integer>> predictedLabels = (List<HashSet<Integer>>) tuningData
					.get(ThresholdTuningDataKeys.predictedLabels);

			@SuppressWarnings("unchecked")
			List<HashSet<Integer>> trueLabels = (List<HashSet<Integer>>) tuningData
					.get(ThresholdTuningDataKeys.trueLabels);

			if (predictedLabels == null || trueLabels == null)
				throw new IllegalArgumentException("Incorrect tuning data. Missing true or predicted labels");

			thresholdsToChange = tuneAndGetAffectedLabels(predictedLabels, trueLabels);
		}
		if (thresholdsToChange == null)
			thresholdsToChange = aThresholdNumerators.keySet();
		HashMap<Integer, Double> sparseThresholds = new HashMap<Integer, Double>();

		for (int label : thresholdsToChange) {
			sparseThresholds.put(label,
					(double) aThresholdNumerators.get(label) / (double) bThresholdDenominators.get(label));
		}

		return sparseThresholds;
	}

	/**
	 * Tunes and returns the set of labels for which thresholds need to be
	 * changed.
	 * 
	 * @param predictedLabels
	 * @param trueLabels
	 * @return Set of labels for which thresholds need to be changed.
	 */
	private HashSet<Integer> tuneAndGetAffectedLabels(List<HashSet<Integer>> predictedLabels,
			List<HashSet<Integer>> trueLabels) {

		HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

		for (int j = 0; j < predictedLabels.size(); j++) {

			HashSet<Integer> predictedPositives = predictedLabels.get(j);
			HashSet<Integer> truePositives = trueLabels.get(j);

			for (int predictedLabel : predictedPositives) {
				bThresholdDenominators.put(predictedLabel, bThresholdDenominators.get(predictedLabel) + 1);
				thresholdsToChange.add(predictedLabel);
			}

			for (int trueLabel : truePositives) {
				if (bThresholdDenominators.containsKey(trueLabel)) {
					bThresholdDenominators.put(trueLabel, bThresholdDenominators.get(trueLabel) + 1);
					thresholdsToChange.add(trueLabel);
					if (predictedPositives.contains(trueLabel))
						aThresholdNumerators.put(trueLabel, aThresholdNumerators.get(trueLabel) + 1);
				}
			}
		}

		return thresholdsToChange;
	}

	@Override
	public double getMacroFmeasure() {
		return computeMacroFmeasure(aThresholdNumerators, bThresholdDenominators);
	}

	private double computeMacroFmeasure(Map<Integer, Integer> aThresholdNumerators,
			Map<Integer, Integer> bThresholdDenominators) {
		return (2.0 / (double) aThresholdNumerators.size()) * aThresholdNumerators.keySet()
				.stream()
				.map(label -> (double) aThresholdNumerators.get(label) / (double) bThresholdDenominators.get(label))
				.reduce(0.0,
						(sum, item) -> sum += item,
						(sum1, sum2) -> sum1 + sum2);
	}

	@Override
	public double getTempMacroFmeasure(Map<String, Object> tuningData) throws Exception {

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> predictedLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDataKeys.predictedLabels);

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> trueLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDataKeys.trueLabels);

		if (predictedLabels == null || trueLabels == null)
			throw new IllegalArgumentException("Incorrect tuning data. Missing true or predicted labels");

		Map<Integer, Integer> aClone = new HashMap<Integer, Integer>();
		Map<Integer, Integer> bClone = new HashMap<Integer, Integer>();
		aClone.putAll(aThresholdNumerators);
		bClone.putAll(bThresholdDenominators);

		for (int j = 0; j < predictedLabels.size(); j++) {

			HashSet<Integer> predictedPositives = predictedLabels.get(j);
			HashSet<Integer> truePositives = trueLabels.get(j);

			for (int predictedLabel : predictedPositives) {
				bClone.put(predictedLabel, bClone.get(predictedLabel) + 1);
			}
			for (int trueLabel : truePositives) {
				if (bClone.containsKey(trueLabel)) {
					bClone.put(trueLabel, bClone.get(trueLabel) + 1);

					if (predictedPositives.contains(trueLabel))
						aClone.put(trueLabel, aClone.get(trueLabel) + 1);
				}
			}
		}

		return computeMacroFmeasure(aClone, bClone);
	}
}