package interfaces;

import java.util.UUID;

import Learner.AbstractLearner;

public interface ILearnerRepository {

	public UUID create(AbstractLearner learner, UUID parentLearnerId);

	public <T extends AbstractLearner> T read(Object learnerId, Class<T> learnerType);

	public AbstractLearner update(Object learnerId, AbstractLearner learner);
}