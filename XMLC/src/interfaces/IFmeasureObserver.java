package interfaces;

import Learner.AbstractLearner;
import event.listeners.IInstanceProcessedListener;

public interface IFmeasureObserver extends IInstanceProcessedListener{
	
	public double getAverageFmeasure(AbstractLearner learner, boolean isPrequential);
	
}