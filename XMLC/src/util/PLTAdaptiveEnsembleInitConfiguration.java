package util;

import interfaces.ILearnerRepository;
import threshold.ThresholdTunerInitOption;
import util.Constants.PLTAdaptiveEnsembleDefaultValues;

public class PLTAdaptiveEnsembleInitConfiguration extends LearnerInitConfiguration {

	private Double alpha;
	private Double epsilon;
	private Integer minTraingInstances;
	private Boolean preferMacroFmeasure;
	private Double retainmentFraction;
	private PLTAdaptiveEnsemblePenalizingStrategies penalizingStrategy;
	private PLTAdaptiveEnsembleAgeFunctions ageFunction;
	private Integer a;
	private Integer c;

	public PLTInitConfiguration individualPLTProperties;
	public ILearnerRepository learnerRepository;
	public ThresholdTunerInitOption tunerInitOption;

	/**
	 * @return the alpha
	 */
	public double getAlpha() {
		return alpha != null ? alpha : PLTAdaptiveEnsembleDefaultValues.alpha;
	}

	/**
	 * @param alpha
	 *            the alpha to set
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * @return the epsilon
	 */
	public double getEpsilon() {
		return epsilon != null ? epsilon : PLTAdaptiveEnsembleDefaultValues.epsilon;
	}

	/**
	 * @param epsilon
	 *            the epsilon to set
	 */
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	/**
	 * @return the minTraingInstances
	 */
	public int getMinTraingInstances() {
		return minTraingInstances != null ? minTraingInstances : PLTAdaptiveEnsembleDefaultValues.minTraingInstances;
	}

	/**
	 * @param minTraingInstances
	 *            the minTraingInstances to set
	 */
	public void setMinTraingInstances(int minTraingInstances) {
		this.minTraingInstances = minTraingInstances;
	}

	/**
	 * @return the preferMacroFmeasure
	 */
	public boolean isPreferMacroFmeasure() {
		return preferMacroFmeasure != null ? preferMacroFmeasure : PLTAdaptiveEnsembleDefaultValues.preferMacroFmeasure;
	}

	/**
	 * @param preferMacroFmeasure
	 *            the preferMacroFmeasure to set
	 */
	public void setPreferMacroFmeasure(boolean preferMacroFmeasure) {
		this.preferMacroFmeasure = preferMacroFmeasure;
	}

	/**
	 * @return the retainmentFraction
	 */
	public double getRetainmentFraction() {
		return retainmentFraction != null ? retainmentFraction : PLTAdaptiveEnsembleDefaultValues.retainmentFraction;
	}

	/**
	 * @param retainmentFraction
	 *            the retainmentFraction to set
	 */
	public void setRetainmentFraction(double retainmentFraction) {
		this.retainmentFraction = retainmentFraction;
	}

	/**
	 * @return the penalizingStrategy
	 */
	public PLTAdaptiveEnsemblePenalizingStrategies getPenalizingStrategy() {
		return penalizingStrategy != null ? penalizingStrategy
				: PLTAdaptiveEnsemblePenalizingStrategies.AgePlusLogOfInverseMacroFm;
	}

	/**
	 * @param penalizingStrategy
	 *            the penalizingStrategy to set
	 */
	public void setPenalizingStrategy(PLTAdaptiveEnsemblePenalizingStrategies penalizingStrategy) {
		this.penalizingStrategy = penalizingStrategy;
	}

	/**
	 * @return the ageFunction
	 */
	public PLTAdaptiveEnsembleAgeFunctions getAgeFunction() {
		return ageFunction != null ? ageFunction : PLTAdaptiveEnsembleAgeFunctions.NumberOfLabelsBased;
	}

	/**
	 * @param ageFunction
	 *            the ageFunction to set
	 */
	public void setAgeFunction(PLTAdaptiveEnsembleAgeFunctions ageFunction) {
		this.ageFunction = ageFunction;
	}

	/**
	 * @return the a
	 */
	public Integer getA() {
		return a != null ? a : PLTAdaptiveEnsembleDefaultValues.a;
	}

	/**
	 * @param a
	 *            the a to set
	 */
	public void setA(Integer a) {
		this.a = a;
	}

	/**
	 * @return the c
	 */
	public Integer getC() {
		return c != null ? c : PLTAdaptiveEnsembleDefaultValues.c;
	}

	/**
	 * @param c
	 *            the c to set
	 */
	public void setC(Integer c) {
		this.c = c;
	}
}