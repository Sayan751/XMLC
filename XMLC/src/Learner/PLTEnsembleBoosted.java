package Learner;

import static java.lang.Math.ceil;
import static java.lang.Math.exp;
import static java.lang.Math.log;

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

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;

import Data.AVPair;
import Data.EstimatePair;
import Data.Instance;
import IO.DataManager;
import interfaces.ILearnerRepository;
import threshold.IAdaptiveTuner;
import threshold.ThresholdTunerFactory;
import threshold.ThresholdTuners;
import util.AdaptivePLTInitConfiguration;
import util.Constants.PLTEnsembleBoostedDefaultValues;
import util.PLTEnsembleBoostedInitConfiguration;
import util.PLTPropertiesForCache;

public class PLTEnsembleBoosted extends AbstractLearner {

	private static final long serialVersionUID = -8401089694679688912L;

	private static Logger logger = LoggerFactory.getLogger(PLTEnsembleBoosted.class);

	transient private ILearnerRepository learnerRepository;

	public ILearnerRepository getLearnerRepository() {
		return learnerRepository;
	}

	public void setLearnerRepository(ILearnerRepository learnerRepository) {
		this.learnerRepository = learnerRepository;
	}

	private List<PLTPropertiesForCache> pltCache;
	private int maxBranchingFactor;
	private int minEpochs;
	private double fZero;
	private boolean isToAggregateByMajorityVote;
	/**
	 * Prefer macro fmeasure for aggregating result.
	 */
	private boolean preferMacroFmeasure;
	private Set<Integer> labelsSeen;
	private int kSlack;

	private AdaptivePLTInitConfiguration pltConfiguration;

	private int ensembleSize;

	public PLTEnsembleBoosted() {
	}

	public PLTEnsembleBoosted(PLTEnsembleBoostedInitConfiguration configuration) {
		super(configuration);

		if (configuration.tunerInitOption == null || configuration.tunerInitOption.aSeed == null
				|| configuration.tunerInitOption.bSeed == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: invalid tuning option; aSeed and bSeed must be provided.");

		learnerRepository = configuration.learnerRepository;
		if (learnerRepository == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: required learnerRepository object is not provided.");

		isToAggregateByMajorityVote = configuration.isToAggregateByMajorityVote();
		preferMacroFmeasure = configuration.isPreferMacroFmeasure();
		fZero = configuration.getfZero();
		minEpochs = configuration.getMinEpochs();
		maxBranchingFactor = configuration.getMaxBranchingFactor();
		kSlack = configuration.getkSlack();

		pltConfiguration = configuration.individualPLTConfiguration;
		pltConfiguration.setToComputeFmeasureOnTopK(isToComputeFmeasureOnTopK);
		pltConfiguration.setDefaultK(defaultK);

		ensembleSize = configuration.getEnsembleSize();
		pltCache = new ArrayList<PLTPropertiesForCache>();

		thresholdTuner = ThresholdTunerFactory.createThresholdTuner(0, ThresholdTuners.AdaptiveOfoFast,
				configuration.tunerInitOption);

		testTuner = ThresholdTunerFactory.createThresholdTuner(0, ThresholdTuners.AdaptiveOfoFast,
				configuration.tunerInitOption);
		testTopKTuner = ThresholdTunerFactory.createThresholdTuner(0, ThresholdTuners.AdaptiveOfoFast,
				configuration.tunerInitOption);

		labelsSeen = new HashSet<Integer>();
	}

	@Override
	public void allocateClassifiers(DataManager data) {

		if (fmeasureObserverAvailable)
			pltConfiguration.fmeasureObserver = fmeasureObserver;

		int[] ks = null;
		if (maxBranchingFactor > PLTEnsembleBoostedDefaultValues.minBranchingFactor) {
			ks = new UniformIntegerDistribution(
					PLTEnsembleBoostedDefaultValues.minBranchingFactor,
					maxBranchingFactor).sample(ensembleSize);
		} else {
			ks = new int[ensembleSize];
			Arrays.fill(ks, PLTEnsembleBoostedDefaultValues.minBranchingFactor);
		}

		double[] alphas = new UniformRealDistribution(PLTEnsembleBoostedDefaultValues.minAlpha,
				PLTEnsembleBoostedDefaultValues.maxAlpha).sample(ensembleSize);

		for (int i = 0; i < ensembleSize; i++) {
			// add random factors
			pltConfiguration.setK(ks[i]);
			pltConfiguration.setAlpha(alphas[i]);

			AdaptivePLT learner = new AdaptivePLT(pltConfiguration);
			learner.allocateClassifiers(data);

			UUID learnerId = learnerRepository.create(learner, getId());
			pltCache.add(new PLTPropertiesForCache(learnerId, learner.m));
		}
	}

	@Override
	public void train(DataManager data) {
		evaluate(data, true);
		int currentDataSetSize = 0;
		while (data.hasNext()) {
			train(data.getNextInstance());
			currentDataSetSize++;
		}
		nTrain += currentDataSetSize;

		tuneThreshold(data);

		evaluate(data, false);
	}

	private void train(Instance instance) {
		int epochs = minEpochs;

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		Set<Integer> truePositive = new HashSet<Integer>(Ints.asList(instance.y));
		ImmutableSet<Integer> diff = Sets.difference(truePositive, labelsSeen)
				.immutableCopy();

		UUID learnerId = null;
		AdaptivePLT learner = null;
		for (PLTPropertiesForCache pltCacheEntry : pltCache) {

			learnerId = pltCacheEntry.learnerId;
			learner = getAdaptivePLT(learnerId);

			learner.train(instance, epochs, true);
			double fm = learner.getFmeasureForInstance(instance, false, false, false);
			epochs = getNextEpochsFromFmeasure(fm);
			logger.info("Current fmeasure: " + fm + ", next epochs: " + epochs);

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

		if (measureTime) {
			getStopwatch().stop();
			totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}
	}

	private AdaptivePLT getAdaptivePLT(UUID learnerId) {
		AdaptivePLT plt = learnerRepository.read(learnerId, AdaptivePLT.class);
		if (plt.fmeasureObserverAvailable) {
			plt.fmeasureObserver = fmeasureObserver;
			plt.addInstanceProcessedListener(fmeasureObserver);
		}
		return plt;
	}

	private int getNextEpochsFromFmeasure(double fm) {
		if (fm == 1)
			return minEpochs;

		int epochs;
		if (fm == 0)
			fm = fZero;

		double alpha = 0.5 * log((1 - fm) / fm);
		epochs = (int) ceil(minEpochs * exp(alpha));

		PoissonDistribution pois = new PoissonDistribution(epochs);
		epochs = pois.sample();

		return epochs < minEpochs ? minEpochs : epochs;
	}

	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> predictions = new HashSet<Integer>();

		pltCache.forEach(
				pltCacheEntry -> predictions.addAll(learnerRepository.read(pltCacheEntry.learnerId, AdaptivePLT.class)
						.getPositiveLabels(x)));

		return predictions;
	}

	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		List<List<EstimatePair>> predictions = getPredictedLabelsAndPosteriorsFromBaseLearners(x, k + kSlack);

		if (!predictions.isEmpty()) {
			Map<Integer, Double> scoreMap = new HashMap<>();
			double fm = 0;

			for (int i = 0; i < ensembleSize; i++) {
				PLTPropertiesForCache cachedPltDetails = pltCache.get(i);

				if (!isToAggregateByMajorityVote) {
					fm = preferMacroFmeasure ? cachedPltDetails.macroFmeasure : cachedPltDetails.avgFmeasure;
				}

				for (EstimatePair pair : predictions.get(i)) {
					int label = pair.getLabel();
					if (!scoreMap.containsKey(label)) {
						scoreMap.put(label, 0.0);
					}

					if (isToAggregateByMajorityVote) {
						// weighted majority vote
						scoreMap.put(label, scoreMap.get(label) + pair.getP());
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
				pltCacheEntry -> predictions.add(learnerRepository.read(pltCacheEntry.learnerId, AdaptivePLT.class)
						.getTopKEstimatesNew(x, k)));
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
