package util;

import interfaces.IFmeasureObserver;
import util.Constants.LearnerDefaultValues;

public class LearnerInitConfiguration {
	private Boolean toComputeFmeasureOnTopK;
	private Integer defaultK;
	public transient IFmeasureObserver fmeasureObserver;
	private Boolean shuffleLabels;
	private Boolean measureTime;

	/**
	 * @return the toComputeFmeasureOnTopK
	 */
	public boolean isToComputeFmeasureOnTopK() {
		return toComputeFmeasureOnTopK != null ? toComputeFmeasureOnTopK
				: LearnerDefaultValues.isToComputeFmeasureOnTopK;
	}

	/**
	 * @param toComputeFmeasureOnTopK
	 *            the toComputeFmeasureOnTopK to set
	 */
	public void setToComputeFmeasureOnTopK(boolean toComputeFmeasureOnTopK) {
		this.toComputeFmeasureOnTopK = toComputeFmeasureOnTopK;
	}

	/**
	 * @return the defaultK
	 */
	public int getDefaultK() {
		return defaultK != null ? defaultK : LearnerDefaultValues.defaultK;
	}

	/**
	 * @param defaultK
	 *            the defaultK to set
	 */
	public void setDefaultK(int defaultK) {
		this.defaultK = defaultK;
	}

	/**
	 * @return the shuffleLabels
	 */
	public boolean isToShuffleLabels() {
		return shuffleLabels != null ? shuffleLabels
				: LearnerDefaultValues.shuffleLabels;
	}

	/**
	 * @param isToShuffleLabels
	 *            the shuffleLabels to set
	 */
	public void setshuffleLabels(boolean isToShuffleLabels) {
		this.shuffleLabels = isToShuffleLabels;
	}

	/**
	 * @return the measureTime
	 */
	public boolean isMeasureTime() {
		return measureTime != null ? measureTime : LearnerDefaultValues.measureTime;
	}

	/**
	 * @param measureTime
	 *            the measureTime to set
	 */
	public void setMeasureTime(boolean measureTime) {
		this.measureTime = measureTime;
	}
}