package event.listeners;

import Learner.AbstractLearner;
import event.args.InstanceProcessedEventArgs;

public interface IInstanceProcessedListener {
	void onInstanceProcessed(AbstractLearner learner, InstanceProcessedEventArgs args);
}