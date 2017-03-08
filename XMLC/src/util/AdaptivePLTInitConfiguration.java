package util;

import util.Constants.AdaptivePLTDefaultValues;

public class AdaptivePLTInitConfiguration extends PLTInitConfiguration {
	private Boolean isToPreferHighestProbLeaf;
	private Double alpha;
	private Boolean isToPreferShallowLeaf;

	/**
	 * @return the isToPreferHighestProbLeaf
	 */
	public boolean isToPreferHighestProbLeaf() {
		return isToPreferHighestProbLeaf != null ? isToPreferHighestProbLeaf
				: AdaptivePLTDefaultValues.isToPreferHighestProbLeaf;
	}

	/**
	 * @param isToPreferHighestProbLeaf
	 *            the isToPreferHighestProbLeaf to set
	 */
	public void setToPreferHighestProbLeaf(boolean isToPreferHighestProbLeaf) {
		this.isToPreferHighestProbLeaf = isToPreferHighestProbLeaf;
	}

	/**
	 * @return the alpha
	 */
	public double getAlpha() {
		return alpha != null ? alpha
				: AdaptivePLTDefaultValues.alpha;
	}

	/**
	 * @param alpha
	 *            the alpha to set
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * @return the isToPreferShallowLeaf
	 */
	public boolean isToPreferShallowLeaf() {
		return isToPreferShallowLeaf != null ? isToPreferShallowLeaf
				: AdaptivePLTDefaultValues.isToPreferShallowLeaf;
	}

	/**
	 * @param isToPreferShallowLeaf
	 *            the isToPreferShallowLeaf to set
	 */
	public void setToPreferShallowLeaf(boolean isToPreferShallowLeaf) {
		this.isToPreferShallowLeaf = isToPreferShallowLeaf;
	}

}