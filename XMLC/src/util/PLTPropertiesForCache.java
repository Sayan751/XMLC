package util;

import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

public class PLTPropertiesForCache {
	public UUID learnerId;
	public int numberOfInstances;
	public int numberOfLabels;
	public double avgFmeasure;
	public double macroFmeasure;
	public Set<Integer> labels;

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
}