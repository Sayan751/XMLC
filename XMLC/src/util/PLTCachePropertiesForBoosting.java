package util;

import java.util.Set;
import java.util.UUID;

public class PLTCachePropertiesForBoosting extends PLTPropertiesForCache {

	public double lambdaCorrect = 0.00002;
	public double lambdaWrong = 0.00001;

	public PLTCachePropertiesForBoosting() {
		super();
	}

	public PLTCachePropertiesForBoosting(UUID learnerId) {
		super(learnerId);
	}

	public PLTCachePropertiesForBoosting(UUID learnerId, int numberOfLabels) {
		super(learnerId, numberOfLabels);
	}

	public PLTCachePropertiesForBoosting(UUID learnerId, Set<Integer> labels) {
		super(learnerId, labels);
	}

	public PLTCachePropertiesForBoosting(UUID learnerId, int numberOfLabels, int numberOfInstances,
			double avgFmeasure) {
		super(learnerId, numberOfLabels, numberOfInstances, avgFmeasure);
	}
}