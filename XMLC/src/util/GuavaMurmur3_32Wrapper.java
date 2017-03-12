package util;

import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;

/**
 * Wraps com.google.common.hash.Murmur3_32HashFunction, so that it can be
 * de/serialized.
 * 
 * @author Sayan
 *
 */
public class GuavaMurmur3_32Wrapper {
	private int seed;
	private transient com.google.common.hash.HashFunction murmur3;

	public GuavaMurmur3_32Wrapper() {
	}

	public GuavaMurmur3_32Wrapper(int seed) {
		this.seed = seed;
	}

	public HashCode hashInt(int input) {
		return getMurmur3_32Hasher().hashInt(input);
	}

	private com.google.common.hash.HashFunction getMurmur3_32Hasher() {
		if (murmur3 == null)
			murmur3 = Hashing.murmur3_32(seed);
		return murmur3;
	}
}
