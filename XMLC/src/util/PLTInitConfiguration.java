package util;

import threshold.ThresholdTunerInitOption;
import threshold.ThresholdTuners;
import util.Constants.PLTDefaultValues;

public class PLTInitConfiguration extends LearnerInitConfiguration {
	private Double gamma;
	private Double lambda;
	private Integer epochs;
	private String hasher;
	private Integer hd;
	private Integer k;
	private String treeType;
	public String treeFile;
	public ThresholdTuners tunerType;
	public ThresholdTunerInitOption tunerInitOption;

	/**
	 * @return the gamma
	 */
	public double getGamma() {
		return gamma != null ? gamma : PLTDefaultValues.gamma;
	}

	/**
	 * @param gamma
	 *            the gamma to set
	 */
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	/**
	 * @return the lambda
	 */
	public double getLambda() {
		return lambda != null ? lambda : PLTDefaultValues.lambda;
	}

	/**
	 * @param lambda
	 *            the lambda to set
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * @return the epochs
	 */
	public int getEpochs() {
		return epochs != null ? epochs : PLTDefaultValues.epochs;
	}

	/**
	 * @param epochs
	 *            the epochs to set
	 */
	public void setEpochs(int epochs) {
		this.epochs = epochs;
	}

	/**
	 * @return the hasher
	 */
	public String getHasher() {
		return hasher == null || hasher.isEmpty() ? PLTDefaultValues.hasher : hasher;
	}

	/**
	 * @param hasher
	 *            the hasher to set
	 */
	public void setHasher(String hasher) {
		this.hasher = hasher;
	}

	/**
	 * @return the hd
	 */
	public int getHd() {
		return hd != null ? hd : PLTDefaultValues.hd;
	}

	/**
	 * @param hd
	 *            the hd to set
	 */
	public void setHd(int hd) {
		this.hd = hd;
	}

	/**
	 * @return the k
	 */
	public int getK() {
		return k != null ? k : PLTDefaultValues.k;
	}

	/**
	 * @param k
	 *            the k to set
	 */
	public void setK(int k) {
		this.k = k;
	}

	/**
	 * @return the treeType
	 */
	public String getTreeType() {
		return treeType == null || treeType.isEmpty() ? PLTDefaultValues.treeType : treeType;
	}

	/**
	 * @param treeType
	 *            the treeType to set
	 */
	public void setTreeType(String treeType) {
		this.treeType = treeType;
	}
}