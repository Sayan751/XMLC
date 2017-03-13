package event.listeners;

import Learner.AbstractLearner;
import event.args.InstanceTestedEventArgs;

public interface IInstanceTestedListener {
	void onInstanceTested(AbstractLearner learner, InstanceTestedEventArgs args);
}
