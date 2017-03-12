package preprocessing;

import java.util.ArrayList;

import com.google.common.hash.Hashing;

import Data.AVPair;
import Data.AVTable;
import util.GuavaMurmur3_32Wrapper;

/**
 * Uses murmur3_32 from guava.
 * 
 * @author Sayan
 *
 */
public class AdaptiveMurmurHasher2 implements FeatureHasher, IAdaptiveHasher {

	private int seed;
	private int nFeatures;
	private ArrayList<GuavaMurmur3_32Wrapper> taskhash;

	public AdaptiveMurmurHasher2() {
	}

	public AdaptiveMurmurHasher2(int seed, int nFeatures, int nTasks) {
		this.nFeatures = nFeatures;
		this.seed = seed;
		this.taskhash = new ArrayList<GuavaMurmur3_32Wrapper>();
		for (int i = 0; i < nTasks; i++) {
			adaptForNewTask();
		}
	}

	@Override
	public void adaptForNewTask() {
		/*only taskhash is changed as tasksign is not used in MurmurHasher to provide the sign.*/
		this.taskhash.add(new GuavaMurmur3_32Wrapper(seed + this.taskhash.size()));
	}

	public int getIndex(int task, int feature) {
		return Hashing.consistentHash(
				this.taskhash.get(task)
						.hashInt(feature),
				nFeatures);
	}

	public int getSign(int label, int feature) {
		int value = (label << 1 - 1) * feature;
		return ((value & 1) == 0) ? -1 : 1;
	}

	@Override
	public AVPair[] transformRowSparse(AVPair[] row) {
		return null;
	}

	@Override
	public AVPair[] transformRowSparse(AVPair[] row, int taskid) {
		return null;
	}

	@Override
	public AVTable transformSparse(AVTable data) {
		return null;
	}
}
