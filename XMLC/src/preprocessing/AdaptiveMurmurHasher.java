package preprocessing;

import util.HashFunction;

public class AdaptiveMurmurHasher extends MurmurHasher implements IAdaptiveHasher {

	private int seed;

	public AdaptiveMurmurHasher() {
	}

	public AdaptiveMurmurHasher(int seed, int nFeatures, int nTasks) {
		super(seed, nFeatures, nTasks);
		this.seed = seed;
	}

	@Override
	public void adaptForNewTask() {
		/*only taskhash is changed as tasksign is not used in MurmurHasher to provide the sign.*/
		this.taskhash.add(new HashFunction(seed + this.taskhash.size(), nFeatures));
	}

}
