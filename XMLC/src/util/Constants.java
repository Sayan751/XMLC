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
		public static final int aSeed = 100;
		public static final int bSeed = 200;
	}

	public static class PLTAdaptiveEnsembleDefaultValues {
		public static final double epsilon = 0.1;
		public static final double retainmentFraction = 0.1;
		public static final int minTraingInstances = 1000;
		public static final double alpha = 0.5;
		public static final boolean preferMacroFmeasure = true;
		public static final int c = 100;
		public static final int a = 3;
	}

	public static class AdaptivePLTDefaultValues {
		public static final boolean isToPreferHighestProbLeaf = true;
		public static final double alpha = 0.5;
		public static final boolean isToPreferShallowLeaf = true;
	}

	public static class LearnerDefaultValues {
		public static final boolean isToComputeFmeasureOnTopK = true;
		public static final int defaultK = 5;
		public static final boolean shuffleLabels = false;
		public static final boolean measureTime = false;
	}

	public static class PLTEnsembleBoostedDefaultValues {
		public static final int ensembleSize = 10;
		public static final int maxBranchingFactor = 5;
		public static final double fZero = 0.001;
		public static final int minEpochs = 30;
		public static final boolean isToAggregateByMajorityVote = false;
		public static final boolean preferMacroFmeasure = true;
		public static final double minAlpha = 0.4;
		public static final double maxAlpha = 1;
	}

	public static class PLTDefaultValues {
		public static final double gamma = 1.0;
		public static final double lambda = 1.0;
		public static final int epochs = 30;
		public static final String hasher = "Mask";
		public static final int hd = 50000000;
		public static final int k = 2;
		public static final String treeType = "Complete";
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
		public static final String penalizingStrategy = "penalizingStrategy";
		public static final String ageFunction = "ageFunction";
		public static final String pltEnsembleC = "pltEnsembleC";
		public static final String pltEnsembleA = "pltEnsembleA";

		// AdaptivePLT init properties
		public static final String isToPreferHighestProbLeaf = "isToPreferHighestProbLeaf";
		public static final String adpativePLTAlpha = "probabilityWeight";
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