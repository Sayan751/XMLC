package Learner;

import static java.lang.Math.ceil;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.max;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import javax.management.RuntimeErrorException;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;
import com.google.common.util.concurrent.AtomicDouble;

import Data.AVPair;
import Data.EstimatePair;
import Data.Instance;
import IO.DataManager;
import interfaces.ILearnerRepository;
import threshold.IAdaptiveTuner;
import threshold.ThresholdTunerFactory;
import threshold.ThresholdTunerInitOption;
import threshold.ThresholdTuners;
import util.AdaptivePLTInitConfiguration;
import util.BoostingStrategy;
import util.Constants.PLTEnsembleBoostedGenericDefaultValues;
import util.PLTCachePropertiesForBoosting;
import util.PLTEnsembleBoostedGenericInitConfiguration;
import util.PLTInitConfiguration;
import util.PLTPropertiesForCache;

public class PLTEnsembleBoostedGeneric<T extends PLT> extends AbstractLearner {

	private static final long serialVersionUID = -8401089694679688912L;

	private static Logger logger = LoggerFactory.getLogger(PLTEnsembleBoostedGeneric.class);

	transient private ILearnerRepository learnerRepository;

	public ILearnerRepository getLearnerRepository() {
		return learnerRepository;
	}

	public void setLearnerRepository(ILearnerRepository learnerRepository) {
		this.learnerRepository = learnerRepository;
	}

	private Class<T> baseLearnerClass;
	private List<PLTPropertiesForCache> pltCache;
	private int maxBranchingFactor;
	private boolean isToAggregateByMajorityVote;
	private boolean isToAggregateByLambdaCW;
	/**
	 * Prefer macro fmeasure for aggregating result.
	 */
	private boolean preferMacroFmeasure;
	private Set<Integer> labelsSeen;
	private int ensembleSize;
	private int minEpochs;
	private double fZero;
	private int kSlack;
	private int numberOfLabels;
	private BoostingStrategy boostingStrategy;

	private PLTInitConfiguration pltConfiguration;

	private boolean isAdaptivePLT;

	private ThresholdTunerInitOption tunerInitOption;

	public PLTEnsembleBoostedGeneric() {
		isAdaptivePLT = baseLearnerClass == AdaptivePLT.class;
	}

	@SuppressWarnings("unchecked")
	public PLTEnsembleBoostedGeneric(PLTEnsembleBoostedGenericInitConfiguration configuration) {
		super(configuration);

		if (configuration.tunerInitOption == null || configuration.tunerInitOption.aSeed == null
				|| configuration.tunerInitOption.bSeed == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: invalid tuning option; aSeed and bSeed must be provided.");

		learnerRepository = configuration.learnerRepository;
		if (learnerRepository == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: required learnerRepository object is not provided.");

		boostingStrategy = configuration.getBoostingStrategy();
		isToAggregateByMajorityVote = configuration.isToAggregateByMajorityVote();
		isToAggregateByLambdaCW = configuration.isToAggregateByLambdaCW();
		if (isToAggregateByLambdaCW && isToAggregateByMajorityVote)
			throw new IllegalArgumentException(
					"Incorrect aggregation configuration; aggregation can be done by majority vote or by lambdas, not both.");

		if (isToAggregateByLambdaCW && !boostingStrategy.equals(BoostingStrategy.threshold))
			throw new IllegalArgumentException(
					"Incorrect boosting strategy; boosting strategy needs to be based on threshold with aggregation by lambdas.");
		System.out.println(getClass().getGenericSuperclass());
		// baseLearnerClass = (Class<T>) ((ParameterizedType)
		// getClass().getGenericSuperclass())
		// .getActualTypeArguments()[0];
		baseLearnerClass = (Class<T>) configuration.getBaseLearnerClass();
		isAdaptivePLT = baseLearnerClass == AdaptivePLT.class;
		if ((isAdaptivePLT && !(configuration.individualPLTConfiguration instanceof AdaptivePLTInitConfiguration)) ||
				(!isAdaptivePLT && !(configuration.individualPLTConfiguration instanceof PLTInitConfiguration)))
			throw new IllegalArgumentException("Incorrect base learner initialization configuratione.");

		preferMacroFmeasure = configuration.isPreferMacroFmeasure();
		fZero = configuration.getfZero();
		maxBranchingFactor = configuration.getMaxBranchingFactor();
		minEpochs = configuration.getMinEpochs();
		kSlack = configuration.getkSlack();

		pltConfiguration = configuration.individualPLTConfiguration;
		pltConfiguration.setToComputeFmeasureOnTopK(isToComputeFmeasureOnTopK);
		pltConfiguration.setDefaultK(defaultK);

		ensembleSize = configuration.getEnsembleSize();
		pltCache = new ArrayList<PLTPropertiesForCache>();

		tunerInitOption = configuration.tunerInitOption;

		if (isAdaptivePLT)
			labelsSeen = new HashSet<Integer>();
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		try {
			if (!isAdaptivePLT)
				numberOfLabels = data.getNumberOfLabels();

			thresholdTuner = ThresholdTunerFactory.createThresholdTuner(isAdaptivePLT ? 0 : numberOfLabels,
					ThresholdTuners.AdaptiveOfoFast, tunerInitOption);

			testTuner = ThresholdTunerFactory.createThresholdTuner(isAdaptivePLT ? 0 : numberOfLabels,
					ThresholdTuners.AdaptiveOfoFast, tunerInitOption);
			testTopKTuner = ThresholdTunerFactory.createThresholdTuner(isAdaptivePLT ? 0 : numberOfLabels,
					ThresholdTuners.AdaptiveOfoFast, tunerInitOption);

			if (fmeasureObserverAvailable)
				pltConfiguration.fmeasureObserver = fmeasureObserver;

			int[] ks = null;
			double[] alphas = null;
			if (maxBranchingFactor > PLTEnsembleBoostedGenericDefaultValues.minBranchingFactor) {
				ks = new UniformIntegerDistribution(
						PLTEnsembleBoostedGenericDefaultValues.minBranchingFactor,
						maxBranchingFactor).sample(ensembleSize);
			} else {
				ks = new int[ensembleSize];
				Arrays.fill(ks, PLTEnsembleBoostedGenericDefaultValues.minBranchingFactor);
			}

			alphas = new UniformRealDistribution(PLTEnsembleBoostedGenericDefaultValues.minAlpha,
					PLTEnsembleBoostedGenericDefaultValues.maxAlpha).sample(ensembleSize);

			for (int i = 0; i < ensembleSize; i++) {
				// add random factors
				pltConfiguration.setK(ks[i]);
				if (isAdaptivePLT)
					((AdaptivePLTInitConfiguration) pltConfiguration).setAlpha(alphas[i]);

				T learner = baseLearnerClass.getConstructor(pltConfiguration.getClass())
						.newInstance(pltConfiguration);
				learner.allocateClassifiers(data);

				UUID learnerId = learnerRepository.create(learner, getId());
				pltCache.add(new PLTCachePropertiesForBoosting(learnerId, learner.m));
			}
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException
				| InvocationTargetException | NoSuchMethodException | SecurityException e) {
			throw new RuntimeErrorException(new Error(e), "Can't allocate classifier.");
		}
	}

	@Override
	public void train(DataManager data) {
		evaluate(data, true);

		while (data.hasNext()) {
			train(data.getNextInstance());
			nTrain++;
		}

		tuneThreshold(data);

		evaluate(data, false);
	}

	private void train(Instance instance) {
		AtomicDouble lambda = new AtomicDouble(minEpochs);
		int epochs = boostingStrategy == BoostingStrategy.fluent
				? minEpochs
				: max((new PoissonDistribution(minEpochs)).sample(), minEpochs);

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		Set<Integer> truePositive = new HashSet<Integer>(Ints.asList(instance.y));
		if (isAdaptivePLT) {
			ImmutableSet<Integer> diff = Sets.difference(truePositive, labelsSeen)
					.immutableCopy();
			if (!diff.isEmpty()) {
				labelsSeen.addAll(truePositive);
				IAdaptiveTuner tuner = (IAdaptiveTuner) thresholdTuner;
				IAdaptiveTuner tstTuner = (IAdaptiveTuner) testTuner;
				IAdaptiveTuner tstTopkTuner = (IAdaptiveTuner) testTopKTuner;
				diff.forEach(label -> {
					tuner.accomodateNewLabel(label);
					tstTuner.accomodateNewLabel(label);
					tstTopkTuner.accomodateNewLabel(label);
				});
			}
		}

		// expected f-measure for random guess
		double randomGuessFm = (double) truePositive.size()
				/ (double) (truePositive.size() + 0.5 * (isAdaptivePLT ? labelsSeen.size() : numberOfLabels));

		UUID learnerId = null;
		T learner = null;

		for (PLTPropertiesForCache pltCacheEntry : pltCache) {

			logger.info("Current epochs: " + epochs);

			learnerId = pltCacheEntry.learnerId;
			learner = getAdaptivePLT(learnerId);

			learner.train(instance, epochs, true);
			double fm = learner.getFmeasureForInstance(instance, false, false, false);
			epochs = getNextEpochsFromFmeasure(fm, randomGuessFm, lambda,
					pltCacheEntry instanceof PLTCachePropertiesForBoosting
							? (PLTCachePropertiesForBoosting) pltCacheEntry : null);
			// post processing
			// Collect and cache required data from plt
			pltCacheEntry.numberOfInstances = learner.getnTrain();

			if (preferMacroFmeasure)
				pltCacheEntry.macroFmeasure = learner.getMacroFmeasure();
			else
				pltCacheEntry.avgFmeasure = learner.getAverageFmeasure(false, isToComputeFmeasureOnTopK);

			pltCacheEntry.numberOfLabels = learner.m;

			// persist all changes happened during the training.
			learnerRepository.update(learnerId, learner);
		}

		if (measureTime) {
			getStopwatch().stop();
			totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}
	}

	private T getAdaptivePLT(UUID learnerId) {
		T plt = learnerRepository.read(learnerId, baseLearnerClass);
		if (plt.fmeasureObserverAvailable) {
			plt.fmeasureObserver = fmeasureObserver;
			plt.addInstanceProcessedListener(fmeasureObserver);
		}
		return plt;
	}

	private int getNextEpochsFromFmeasure(double fm, double randomGuessFm, AtomicDouble lambda,
			PLTCachePropertiesForBoosting pltCacheEntry) {
		int retVal = 0;
		switch (boostingStrategy) {
		case fluent:

			if (fm == 1)
				return minEpochs;

			int epochs;
			if (fm == 0)
				fm = fZero;

			double alpha = 0.5 * log((1 - fm) / fm);
			epochs = (int) ceil(minEpochs * exp(alpha));

			PoissonDistribution pois = new PoissonDistribution(epochs);
			epochs = pois.sample();

			retVal = epochs < minEpochs ? minEpochs : epochs;

		case threshold:
			if (fm > randomGuessFm) {
				pltCacheEntry.lambdaCorrect += lambda.get() * fm;
				lambda.set(lambda.get() * ((pltCacheEntry.lambdaCorrect + pltCacheEntry.lambdaWrong)
						/ (2 * pltCacheEntry.lambdaCorrect)));
			} else {
				pltCacheEntry.lambdaWrong += lambda.get() * (1 - fm);
				lambda.set(lambda.get() * ((pltCacheEntry.lambdaCorrect + pltCacheEntry.lambdaWrong)
						/ (2 * pltCacheEntry.lambdaWrong)));
			}

			logger.info("Current fmeasure: " + fm + "(" + ((fm > randomGuessFm) ? "above" : "below")
					+ " threshold), lambda: "
					+ lambda + ", lambda_sc: "
					+ pltCacheEntry.lambdaCorrect + ", lambda_sw: " + pltCacheEntry.lambdaWrong);
			retVal = max((new PoissonDistribution(minEpochs)).sample(), minEpochs);

		}
		return retVal;
	}

	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> predictions = new HashSet<Integer>();

		pltCache.forEach(
				pltCacheEntry -> predictions.addAll(learnerRepository.read(pltCacheEntry.learnerId, baseLearnerClass)
						.getPositiveLabels(x)));

		return predictions;
	}

	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		List<List<EstimatePair>> predictions = getPredictedLabelsAndPosteriorsFromBaseLearners(x, k + kSlack);

		if (!predictions.isEmpty()) {
			Map<Integer, Double> scoreMap = new HashMap<>();
			double logLambda = 0, fm = 0;

			for (int i = 0; i < ensembleSize; i++) {
				PLTPropertiesForCache cachedPltDetails = pltCache.get(i);

				if (isToAggregateByLambdaCW) {
					logLambda = log(((PLTCachePropertiesForBoosting) cachedPltDetails).lambdaCorrect
							/ ((PLTCachePropertiesForBoosting) cachedPltDetails).lambdaWrong);
				} else if (!(isToAggregateByLambdaCW || isToAggregateByMajorityVote)) {
					fm = preferMacroFmeasure ? cachedPltDetails.macroFmeasure : cachedPltDetails.avgFmeasure;
				}

				for (EstimatePair pair : predictions.get(i)) {
					int label = pair.getLabel();
					if (!scoreMap.containsKey(label)) {
						scoreMap.put(label, 0.0);
					}

					if (isToAggregateByMajorityVote) {
						scoreMap.put(label, scoreMap.get(label) + pair.getP());
					} else if (isToAggregateByLambdaCW) {
						scoreMap.put(label, scoreMap.get(label) + (logLambda * pair.getP()));
					} else {
						scoreMap.put(label, scoreMap.get(label) + (fm * pair.getP()));
					}
				}
			}

			// scoreMap.replaceAll((label, score) -> score / ensembleSize);

			return scoreMap.entrySet()
					.stream()
					.sorted(Entry.<Integer, Double>comparingByValue()
							.reversed())
					.limit(k)
					.mapToInt(entry -> entry.getKey())
					.toArray();
		}

		return new int[0];
	}

	private List<List<EstimatePair>> getPredictedLabelsAndPosteriorsFromBaseLearners(AVPair[] x, int k) {
		List<List<EstimatePair>> predictions = new ArrayList<>();

		pltCache.forEach(
				pltCacheEntry -> predictions.add(learnerRepository.read(pltCacheEntry.learnerId, baseLearnerClass)
						.getTopKEstimatesComplete(x, k)));
		return predictions;
	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		return 0;
	}

	@Override
	protected void tuneThreshold(DataManager data) {
		thresholdTuner.getTunedThresholdsSparse(createTuningData(data));
	}
}
