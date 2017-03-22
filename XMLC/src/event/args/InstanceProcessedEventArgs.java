package event.args;

import Data.Instance;

public class InstanceProcessedEventArgs {

	public Instance instance;
	public boolean isPrequential;
	public double fmeasure;
	public double topkFmeasure;

	public InstanceProcessedEventArgs(Instance instance, double fmeasure, double topkFmeasure,
			boolean isPrequential) {
		this.instance = instance;
		this.fmeasure = fmeasure;
		this.topkFmeasure = topkFmeasure;
		this.isPrequential = isPrequential;
	}
}
