package util;

public enum PLTAdaptiveEnsemblePenalizingStrategies {
	/**
	 * Note: this works for both average fmeasure, and macro fmeasure.
	 */
	FmMinusRatioOfInstances,
	/**
	 * Note: this does not work if average fmeasure is preferred over macro
	 * fmeasure.
	 */
	AgePlusLogOfInverseMacroFm
}