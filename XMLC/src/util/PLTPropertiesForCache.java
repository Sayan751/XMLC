package util;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import Data.AVPair;

public class PLTPropertiesForCache {
	public UUID learnerId;
	public int numberOfInstances;
	public int numberOfLabels;
	public double avgFmeasure;
	public double macroFmeasure;
	public Set<Integer> labels;
	/**
	 * Only meant for temporary caching the predictions, and not for perpetual
	 * storing.
	 */
	public Map<AVPair[], int[]> tempTopkPredictions = new HashMap<>();
	/**
	 * Only meant for temporary caching the predictions, and not for perpetual
	 * storing.
	 */
	public Map<AVPair[], HashSet<Integer>> tempPredictions = new HashMap<>();

	public PLTPropertiesForCache(UUID learnerId) {
		this.learnerId = learnerId;
	}

	public PLTPropertiesForCache(UUID learnerId, int numberOfLabels) {
		this(learnerId);
		this.numberOfLabels = numberOfLabels;
	}

	public PLTPropertiesForCache(UUID learnerId, Set<Integer> labels) {
		this(learnerId, labels.size());
		this.labels = new HashSet<Integer>();
		this.labels.addAll(labels);
	}

	public PLTPropertiesForCache(UUID learnerId, int numberOfLabels, int numberOfInstances,
			double avgFmeasure) {
		this(learnerId, numberOfLabels);
		this.numberOfInstances = numberOfInstances;
		this.avgFmeasure = avgFmeasure;
	}

	public void clearAllPredictions() {
		tempPredictions.clear();
		tempTopkPredictions.clear();
	}
}