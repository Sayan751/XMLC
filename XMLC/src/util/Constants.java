package util;

/**
 * Contains all constant values (magic strings) used in this library
 * 
 * @author Sayan
 *
 */
public class Constants {
	/**
	 * Contains constant string literals for the dictionary keys used for
	 * threshold tuning.
	 * 
	 * @author Sayan
	 *
	 */
	public static class ThresholdTuningDataKeys {
		public static final String trueLabels = "trueLabels";
		public static final String predictedLabels = "predictedLabels";
	}

	public static class OFODefaultValues {
		public static final int aSeed = 50;
		public static final int bSeed = 100;
	}

	public static class PLTEnsembleDefaultValues {
		public static final double epsilon = 0.1;
		public static final double retainmentFraction = 0.1;
		public static final int minTraingInstances = 1000;
		public static final double alpha = 0.5;
		public static final String preferMacroFmeasure = "true";
	}

	public static class AdaptivePLTDefaultValues {
		public static final String isToPreferHighestProbLeaf = "true";
		public static final String probabilityWeight = "0.5";
		public static final String isToPreferShallowLeaf = "true";
	}

	public static class LearnerDefaultValues {
		public static final boolean isToComputeFmeasureOnTopK = true;
		public static final int defaultK = 5;
		public static final String shuffleLabels = "false";
	}

	public static class PLTEnsembleBoostedDefaultValues {
		public static final String pltEnsembleBoostedSize = "10";
		public static final String maxBranchingFactor = "5";
		public static final String fZero = "0.001";
		public static final String minEpochs = "30";
		public static final String isToAggregateByMajorityVote = "false";
		public static final String preferMacroFmeasure = "true";
	}

	public static class LearnerInitProperties {
		public static final String individualPLTProperties = "individualPLTProperties";
		public static final String minTraingInstances = "minTraingInstances";
		public static final String isToComputeFmeasureOnTopK = "isToComputeFmeasureOnTopK";
		public static final String defaultK = "defaultK";
		public static final String fmeasureObserver = "fmeasureObserver";
		public static final String tunerType = "tunerType";
		public static final String tunerInitOption = "tunerInitOption";
		public static final String learnerRepository = "learnerRepository";

		// Ensemble PLT init properties
		public static final String pltEnsembleAlpha = "pltEnsembleAlpha";
		public static final String pltEnsembleEpsilon = "pltEnsembleEpsilon";
		public static final String pltEnsembleRetainmentFraction = "pltEnsembleRetainmentFraction";

		// AdaptivePLT init properties
		public static final String isToPreferHighestProbLeaf = "isToPreferHighestProbLeaf";
		public static final String probabilityWeight = "probabilityWeight";
		public static final String isToPreferShallowLeaf = "isToPreferShallowLeaf";

		// Boosted Ensemble PLT init properties
		public static final String pltEnsembleBoostedSize = "pltEnsembleBoostedSize";
		public static final String maxBranchingFactor = "maxBranchingFactor";
		public static final String fZero = "fZero";
		public static final String minEpochs = "minEpoch";
		public static final String shuffleLabels = "shuffleLabels";
		public static final String isToAggregateByMajorityVote = "isToAggregateByMajorityVote";
		public static final String preferMacroFmeasure = "preferMacroFmeasure";
	}
}