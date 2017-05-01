package Learner;

import static java.lang.Math.log;
import static java.lang.Math.max;

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
import util.Constants.PLTEnsembleBoostedWithThresholdDefaultValues;
import util.PLTCachePropertiesForBoosting;
import util.PLTEnsembleBoostedWithThresholdInitConfiguration;

public class PLTEnsembleBoostedWithThreshold extends AbstractLearner {

	private static final long serialVersionUID = -8401089694679688912L;

	private static Logger logger = LoggerFactory.getLogger(PLTEnsembleBoostedWithThreshold.class);

	transient private ILearnerRepository learnerRepository;

	public ILearnerRepository getLearnerRepository() {
		return learnerRepository;
	}

	public void setLearnerRepository(ILearnerRepository learnerRepository) {
		this.learnerRepository = learnerRepository;
	}

	private List<PLTCachePropertiesForBoosting> pltCache;
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
	private int kSlack;

	private AdaptivePLTInitConfiguration pltConfiguration;

	public PLTEnsembleBoostedWithThreshold() {
	}

	public PLTEnsembleBoostedWithThreshold(PLTEnsembleBoostedWithThresholdInitConfiguration configuration) {
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
		isToAggregateByLambdaCW = configuration.isToAggregateByLambdaCW();
		if (isToAggregateByLambdaCW && isToAggregateByMajorityVote)
			throw new IllegalArgumentException(
					"Incorrect aggregation configuration; aggregation can be done by majority vote or by lambdas, not both.");

		preferMacroFmeasure = configuration.isPreferMacroFmeasure();
		maxBranchingFactor = configuration.getMaxBranchingFactor();
		minEpochs = configuration.getMinEpochs();
		kSlack = configuration.getkSlack();

		pltConfiguration = configuration.individualPLTConfiguration;
		pltConfiguration.setToComputeFmeasureOnTopK(isToComputeFmeasureOnTopK);
		pltConfiguration.setDefaultK(defaultK);

		ensembleSize = configuration.getEnsembleSize();
		pltCache = new ArrayList<PLTCachePropertiesForBoosting>();

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
		double[] alphas = null;
		if (maxBranchingFactor > PLTEnsembleBoostedWithThresholdDefaultValues.minBranchingFactor) {
			ks = new UniformIntegerDistribution(
					PLTEnsembleBoostedWithThresholdDefaultValues.minBranchingFactor,
					maxBranchingFactor).sample(ensembleSize);
		} else {
			ks = new int[ensembleSize];
			Arrays.fill(ks, PLTEnsembleBoostedWithThresholdDefaultValues.minBranchingFactor);
		}

		alphas = new UniformRealDistribution(PLTEnsembleBoostedWithThresholdDefaultValues.minAlpha,
				PLTEnsembleBoostedWithThresholdDefaultValues.maxAlpha).sample(ensembleSize);
		for (int i = 0; i < ensembleSize; i++) {
			// add random factors
			pltConfiguration.setK(ks[i]);
			pltConfiguration.setAlpha(alphas[i]);

			AdaptivePLT learner = new AdaptivePLT(pltConfiguration);
			learner.allocateClassifiers(data);

			UUID learnerId = learnerRepository.create(learner, getId());
			pltCache.add(new PLTCachePropertiesForBoosting(learnerId, learner.m));
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
		double lambda = minEpochs;
		int epochs;

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		Set<Integer> truePositive = new HashSet<Integer>(Ints.asList(instance.y));
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

		// expected f-measure for random guess
		double randomGuessFm = (double) truePositive.size() / (double) (truePositive.size() + 0.5 * labelsSeen.size());

		UUID learnerId = null;
		AdaptivePLT learner = null;

		for (PLTCachePropertiesForBoosting pltCacheEntry : pltCache) {

			epochs = max((new PoissonDistribution(lambda)).sample(), minEpochs);

			logger.info("Current epochs: " + epochs);

			learnerId = pltCacheEntry.learnerId;
			learner = getAdaptivePLT(learnerId);

			learner.train(instance, epochs, true);
			double fm = learner.getFmeasureForInstance(instance, false, false, false);

			if (fm > randomGuessFm) {
				pltCacheEntry.lambdaCorrect += lambda * fm;
				lambda = lambda * ((pltCacheEntry.lambdaCorrect + pltCacheEntry.lambdaWrong)
						/ (2 * pltCacheEntry.lambdaCorrect));
			} else {
				pltCacheEntry.lambdaWrong += lambda * (1 - fm);
				lambda = lambda * ((pltCacheEntry.lambdaCorrect + pltCacheEntry.lambdaWrong)
						/ (2 * pltCacheEntry.lambdaWrong));
			}

			logger.info("Current fmeasure: " + fm + "(" + ((fm > randomGuessFm) ? "above" : "below")
					+ " threshold), lambda: "
					+ lambda + ", lambda_sc: "
					+ pltCacheEntry.lambdaCorrect + ", lambda_sw: " + pltCacheEntry.lambdaWrong);

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

	private AdaptivePLT getAdaptivePLT(UUID learnerId) {
		AdaptivePLT plt = learnerRepository.read(learnerId, AdaptivePLT.class);
		if (plt.fmeasureObserverAvailable) {
			plt.fmeasureObserver = fmeasureObserver;
			plt.addInstanceProcessedListener(fmeasureObserver);
		}
		return plt;
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
			double logLambda = 0, fm = 0;

			for (int i = 0; i < ensembleSize; i++) {
				PLTCachePropertiesForBoosting cachedPltDetails = pltCache.get(i);

				if (isToAggregateByLambdaCW) {
					logLambda = log(cachedPltDetails.lambdaCorrect / cachedPltDetails.lambdaWrong);
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
