package event.listeners;

import Learner.AbstractLearner;

public interface IFmeasureObserver extends IInstanceProcessedListener{
	
	public double getAverageFmeasure(AbstractLearner learner, boolean isPrequential);
	
}