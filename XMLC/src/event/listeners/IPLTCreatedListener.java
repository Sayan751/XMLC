package event.listeners;

import event.args.PLTCreationEventArgs;

public interface IPLTCreatedListener {
	void onPLTCreated(Object source, PLTCreationEventArgs args);
}