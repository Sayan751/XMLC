package event.listeners;

import event.args.PLTDiscardedEventArgs;

public interface IPLTDiscardedListener {
	void onPLTDiscarded(Object source, PLTDiscardedEventArgs args);
}