package Data;

import java.util.Arrays;

public class Instance {
	public int[] y;
	public AVPair[] x;

	public Instance(AVPair[] x, int[] y) {
		this.x = x;
		this.y = y;
	}

	@Override
	public String toString() {
		return "{x:" + Arrays.toString(x) + ", y:" + Arrays.toString(y)+"}";
	}

}
