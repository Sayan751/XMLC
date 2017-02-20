package threshold;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

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

	public AdaptiveOfoFastThresholdTuner(int numberOfLabels, ThresholdTunerInitOption thresholdTunerInitOption) {
		super(numberOfLabels, thresholdTunerInitOption);

		logger.info("#####################################################");
		logger.info("#### OFO Fast");
		logger.info("#### numberOfLabels: " + numberOfLabels);

		aSeed = thresholdTunerInitOption != null && thresholdTunerInitOption.aSeed != null
				? thresholdTunerInitOption.aSeed : OFODefaultValues.aSeed;

		bSeed = thresholdTunerInitOption != null && thresholdTunerInitOption.bSeed != null
				? thresholdTunerInitOption.bSeed : OFODefaultValues.bSeed;

		aThresholdNumerators = new HashMap<Integer, Integer>();
		bThresholdDenominators = new HashMap<Integer, Integer>();

		int[] aInit = thresholdTunerInitOption.aInit;
		int[] bInit = thresholdTunerInitOption.bInit;

		if (thresholdTunerInitOption != null
				&& aInit != null && aInit.length > 0 && aInit.length == numberOfLabels
				&& bInit != null && bInit.length > 0 && bInit.length == numberOfLabels) {

			IntStream.range(0, numberOfLabels)
					.forEach(label -> accomodateNewLabel(label, aInit[label], bInit[label]));

			logger.info("#### a[] and b[] are initialized with predefined values");

		} else {

			IntStream.range(0, numberOfLabels)
					.forEach(label -> accomodateNewLabel(label));

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

		if (tuningData == null)
			throw new IllegalArgumentException("Incorrect tuning data");

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> predictedLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDataKeys.predictedLabels);

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> trueLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDataKeys.trueLabels);

		if (predictedLabels == null || trueLabels == null)
			throw new IllegalArgumentException("Incorrect tuning data. Missing true or predicted labels");

		HashSet<Integer> thresholdsToChange = tuneAndGetAffectedLabels(predictedLabels, trueLabels);
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
				if (trueLabel < numberOfLabels) {
					bThresholdDenominators.put(trueLabel, bThresholdDenominators.get(trueLabel) + 1);
					thresholdsToChange.add(trueLabel);
					if (predictedPositives.contains(trueLabel))
						aThresholdNumerators.put(trueLabel, bThresholdDenominators.get(trueLabel) + 1);
				}
			}
		}

		return thresholdsToChange;
	}
}