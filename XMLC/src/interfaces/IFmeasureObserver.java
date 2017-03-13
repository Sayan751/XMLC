package interfaces;

import Learner.AbstractLearner;
import event.listeners.IInstanceProcessedListener;
import event.listeners.IInstanceTestedListener;

public interface IFmeasureObserver extends IInstanceProcessedListener, IInstanceTestedListener {

	public double getAverageFmeasure(AbstractLearner learner, boolean isPrequential);

	public double getTestAverageFmeasure(AbstractLearner learner, boolean isTopk);
}