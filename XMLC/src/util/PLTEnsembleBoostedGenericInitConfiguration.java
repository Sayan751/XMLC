package util;

import interfaces.ILearnerRepository;
import threshold.ThresholdTunerInitOption;
import util.Constants.PLTEnsembleBoostedGenericDefaultValues;

public class PLTEnsembleBoostedGenericInitConfiguration extends LearnerInitConfiguration {
	private Integer ensembleSize;
	private Double fZero;
	private Boolean isToAggregateByMajorityVote;
	private Boolean isToAggregateByLambdaCW;
	private Integer maxBranchingFactor;
	private Boolean preferMacroFmeasure;
	private Integer minEpochs;
	private Integer kSlack;
	private BoostingStrategy boostingStrategy;
	private Class<?> baseLearnerClass;

	public PLTInitConfiguration individualPLTConfiguration;
	public transient ILearnerRepository learnerRepository;
	public ThresholdTunerInitOption tunerInitOption;

	/**
	 * @return the ensembleSize
	 */
	public int getEnsembleSize() {
		return ensembleSize != null ? ensembleSize : PLTEnsembleBoostedGenericDefaultValues.ensembleSize;
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
		return fZero != null ? fZero : PLTEnsembleBoostedGenericDefaultValues.fZero;
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
		return maxBranchingFactor != null ? maxBranchingFactor
				: PLTEnsembleBoostedGenericDefaultValues.maxBranchingFactor;
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
				: PLTEnsembleBoostedGenericDefaultValues.preferMacroFmeasure;
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
				: PLTEnsembleBoostedGenericDefaultValues.isToAggregateByMajorityVote;
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
				: PLTEnsembleBoostedGenericDefaultValues.isToAggregateByLambdaCW;
	}

	public void setToAggregateByLambdaCW(boolean isToAggregateByLambdaCW) {
		this.isToAggregateByLambdaCW = isToAggregateByLambdaCW;
	}

	/**
	 * @return the minEpochs
	 */
	public int getMinEpochs() {
		return minEpochs != null ? minEpochs : PLTEnsembleBoostedGenericDefaultValues.minEpochs;
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
		return kSlack != null ? kSlack : PLTEnsembleBoostedGenericDefaultValues.kSlack;
	}

	/**
	 * @param kSlack
	 *            the kSlack to set
	 */
	public void setkSlack(int kSlack) {
		this.kSlack = kSlack;
	}

	/**
	 * @return the boostingStrategy
	 */
	public BoostingStrategy getBoostingStrategy() {
		return boostingStrategy != null ? boostingStrategy : PLTEnsembleBoostedGenericDefaultValues.boostingStrategy;
	}

	/**
	 * @param boostingStrategy
	 *            the boostingStrategy to set
	 */
	public void setBoostingStrategy(BoostingStrategy boostingStrategy) {
		this.boostingStrategy = boostingStrategy;
	}

	/**
	 * @return the baseLearnerClass
	 */
	public Class<?> getBaseLearnerClass() {
		return baseLearnerClass;
	}

	/**
	 * @param baseLearnerClass the baseLearnerClass to set
	 */
	public void setBaseLearnerClass(Class<?> baseLearnerClass) {
		this.baseLearnerClass = baseLearnerClass;
	}
}