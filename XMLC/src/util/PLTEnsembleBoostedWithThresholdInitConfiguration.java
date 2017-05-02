package util;

import interfaces.ILearnerRepository;
import threshold.ThresholdTunerInitOption;
import util.Constants.PLTEnsembleBoostedWithThresholdDefaultValues;

public class PLTEnsembleBoostedWithThresholdInitConfiguration extends LearnerInitConfiguration {
	private Integer ensembleSize;
	private Boolean isToAggregateByMajorityVote;
	private Boolean isToAggregateByLambdaCW;
	private Integer maxBranchingFactor;
	private Boolean preferMacroFmeasure;
	private Integer minEpochs;
	private Integer kSlack;

	public AdaptivePLTInitConfiguration individualPLTConfiguration;
	public transient ILearnerRepository learnerRepository;
	public ThresholdTunerInitOption tunerInitOption;

	/**
	 * @return the ensembleSize
	 */
	public int getEnsembleSize() {
		return ensembleSize != null ? ensembleSize : PLTEnsembleBoostedWithThresholdDefaultValues.ensembleSize;
	}

	/**
	 * @param ensembleSize
	 *            the ensembleSize to set
	 */
	public void setEnsembleSize(int ensembleSize) {
		this.ensembleSize = ensembleSize;
	}

	/**
	 * @return the maxBranchingFactor
	 */
	public int getMaxBranchingFactor() {
		return maxBranchingFactor != null ? maxBranchingFactor
				: PLTEnsembleBoostedWithThresholdDefaultValues.maxBranchingFactor;
	}

	/**
	 * @param maxBranchingFactor
	 *            the maxBranchingFactor to set
	 */
	public void setMaxBranchingFactor(int maxBranchingFactor) {
		this.maxBranchingFactor = maxBranchingFactor;
	}

	/**
	 * @return the preferMacroFmeasure
	 */
	public boolean isPreferMacroFmeasure() {
		return preferMacroFmeasure != null ? preferMacroFmeasure
				: PLTEnsembleBoostedWithThresholdDefaultValues.preferMacroFmeasure;
	}

	/**
	 * @param preferMacroFmeasure
	 *            the preferMacroFmeasure to set
	 */
	public void setPreferMacroFmeasure(boolean preferMacroFmeasure) {
		this.preferMacroFmeasure = preferMacroFmeasure;
	}

	/**
	 * @return the isToAggregateByMajorityVote
	 */
	public boolean isToAggregateByMajorityVote() {
		return isToAggregateByMajorityVote != null ? isToAggregateByMajorityVote
				: PLTEnsembleBoostedWithThresholdDefaultValues.isToAggregateByMajorityVote;
	}

	/**
	 * @param isToAggregateByMajorityVote
	 *            the isToAggregateByMajorityVote to set
	 */
	public void setToAggregateByMajorityVote(boolean isToAggregateByMajorityVote) {
		this.isToAggregateByMajorityVote = isToAggregateByMajorityVote;
	}

	public boolean isToAggregateByLambdaCW() {
		return isToAggregateByLambdaCW != null ? isToAggregateByLambdaCW
				: PLTEnsembleBoostedWithThresholdDefaultValues.isToAggregateByLambdaCW;
	}

	public void setToAggregateByLambdaCW(boolean isToAggregateByLambdaCW) {
		this.isToAggregateByLambdaCW = isToAggregateByLambdaCW;
	}

	/**
	 * @return the minEpochs
	 */
	public int getMinEpochs() {
		return minEpochs != null ? minEpochs : PLTEnsembleBoostedWithThresholdDefaultValues.minEpochs;
	}

	/**
	 * @param minEpochs
	 *            the defaultEpochs to set
	 */
	public void setMinEpochs(int minEpochs) {
		this.minEpochs = minEpochs;
	}

	/**
	 * @return the kSlack
	 */
	public int getkSlack() {
		return kSlack != null ? kSlack : PLTEnsembleBoostedWithThresholdDefaultValues.kSlack;
	}

	/**
	 * @param kSlack
	 *            the kSlack to set
	 */
	public void setkSlack(int kSlack) {
		this.kSlack = kSlack;
	}
}