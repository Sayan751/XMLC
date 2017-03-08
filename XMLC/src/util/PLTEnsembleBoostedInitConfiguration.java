package util;

import interfaces.ILearnerRepository;
import threshold.ThresholdTunerInitOption;
import util.Constants.PLTEnsembleBoostedDefaultValues;

public class PLTEnsembleBoostedInitConfiguration extends LearnerInitConfiguration {
	private Integer ensembleSize;
	private Double fZero;
	private Boolean isToAggregateByMajorityVote;
	private Integer maxBranchingFactor;
	private Integer minEpochs;
	private Boolean preferMacroFmeasure;

	public AdaptivePLTInitConfiguration individualPLTConfiguration;
	public ILearnerRepository learnerRepository;
	public ThresholdTunerInitOption tunerInitOption;

	/**
	 * @return the ensembleSize
	 */
	public int getEnsembleSize() {
		return ensembleSize != null ? ensembleSize : PLTEnsembleBoostedDefaultValues.ensembleSize;
	}

	/**
	 * @param ensembleSize
	 *            the ensembleSize to set
	 */
	public void setEnsembleSize(int ensembleSize) {
		this.ensembleSize = ensembleSize;
	}

	/**
	 * @return the fZero
	 */
	public double getfZero() {
		return fZero != null ? fZero : PLTEnsembleBoostedDefaultValues.fZero;
	}

	/**
	 * @param fZero
	 *            the fZero to set
	 */
	public void setfZero(double fZero) {
		this.fZero = fZero;
	}

	/**
	 * @return the maxBranchingFactor
	 */
	public int getMaxBranchingFactor() {
		return maxBranchingFactor != null ? maxBranchingFactor : PLTEnsembleBoostedDefaultValues.maxBranchingFactor;
	}

	/**
	 * @param maxBranchingFactor
	 *            the maxBranchingFactor to set
	 */
	public void setMaxBranchingFactor(int maxBranchingFactor) {
		this.maxBranchingFactor = maxBranchingFactor;
	}

	/**
	 * @return the minEpochs
	 */
	public int getMinEpochs() {
		return minEpochs != null ? minEpochs : PLTEnsembleBoostedDefaultValues.minEpochs;
	}

	/**
	 * @param minEpochs
	 *            the minEpochs to set
	 */
	public void setMinEpochs(int minEpochs) {
		this.minEpochs = minEpochs;
	}

	/**
	 * @return the preferMacroFmeasure
	 */
	public boolean isPreferMacroFmeasure() {
		return preferMacroFmeasure != null ? preferMacroFmeasure
				: PLTEnsembleBoostedDefaultValues.preferMacroFmeasure;
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
				: PLTEnsembleBoostedDefaultValues.isToAggregateByMajorityVote;
	}

	/**
	 * @param isToAggregateByMajorityVote
	 *            the isToAggregateByMajorityVote to set
	 */
	public void setToAggregateByMajorityVote(boolean isToAggregateByMajorityVote) {
		this.isToAggregateByMajorityVote = isToAggregateByMajorityVote;
	}
}