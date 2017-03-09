package Learner;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.round;
import static java.lang.Math.toIntExact;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.common.primitives.Ints;

import Data.AVPair;
import Data.Instance;
import IO.DataManager;
import interfaces.ILearnerRepository;
import threshold.IAdaptiveTuner;
import threshold.ThresholdTunerFactory;
import threshold.ThresholdTuners;
import util.AdaptivePLTInitConfiguration;
import util.LearnerInitConfiguration;
import util.PLTEnsembleBoostedInitConfiguration;
import util.PLTPropertiesForCache;

public class PLTEnsembleBoosted extends AbstractLearner {

	private static final long serialVersionUID = -8401089694679688912L;

	private static Logger logger = LoggerFactory.getLogger(PLTEnsembleBoosted.class);

	transient private ILearnerRepository learnerRepository;
	private List<PLTPropertiesForCache> pltCache;
	private int maxBranchingFactor;
	private int minEpochs;
	private double fZero;
	private boolean isToAggregateByMajorityVote;
	/**
	 * Prefer macro fmeasure for aggregating result.
	 */
	private boolean preferMacroFmeasure;
	private static Random random;
	private Set<Integer> labelsSeen;

	public PLTEnsembleBoosted() {
	}

	public PLTEnsembleBoosted(LearnerInitConfiguration configuration) {
		super(configuration);

		PLTEnsembleBoostedInitConfiguration ensembleConfiguration = configuration instanceof PLTEnsembleBoostedInitConfiguration
				? (PLTEnsembleBoostedInitConfiguration) configuration : null;
		if (ensembleConfiguration == null)
			throw new IllegalArgumentException("Invalid init configuration.");

		if (ensembleConfiguration.tunerInitOption == null || ensembleConfiguration.tunerInitOption.aSeed == null
				|| ensembleConfiguration.tunerInitOption.bSeed == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: invalid tuning option; aSeed and bSeed must be provided.");

		learnerRepository = ensembleConfiguration.learnerRepository;
		if (learnerRepository == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: required learnerRepository object is not provided.");

		random = new Random();

		isToAggregateByMajorityVote = ensembleConfiguration.isToAggregateByMajorityVote();
		preferMacroFmeasure = ensembleConfiguration.isPreferMacroFmeasure();
		fZero = ensembleConfiguration.getfZero();
		minEpochs = ensembleConfiguration.getMinEpochs();
		maxBranchingFactor = ensembleConfiguration.getMaxBranchingFactor();

		AdaptivePLTInitConfiguration pltConfiguration = ensembleConfiguration.individualPLTConfiguration;
		pltConfiguration.setToComputeFmeasureOnTopK(isToComputeFmeasureOnTopK);
		pltConfiguration.setDefaultK(defaultK);
		if (fmeasureObserverAvailable)
			pltConfiguration.fmeasureObserver = fmeasureObserver;
		pltConfiguration.setshuffleLabels(shuffleLabels);

		int ensembleSize = ensembleConfiguration.getEnsembleSize();
		pltCache = new ArrayList<PLTPropertiesForCache>();
		IntStream.range(0, ensembleSize)
				.forEach(i -> {
					try {
						addNewPLT(pltConfiguration);
					} catch (Exception e) {
						throw new RuntimeException(e);
					}
				});

		logger.info("Ensemble size:" + pltCache.size());

		thresholdTuner = ThresholdTunerFactory.createThresholdTuner(1, ThresholdTuners.AdaptiveOfoFast,
				ensembleConfiguration.tunerInitOption);

		labelsSeen = new HashSet<Integer>(Arrays.asList(0));
	}

	private void addNewPLT(AdaptivePLTInitConfiguration pltConfiguration) {

		// add random branching factor
		int k = maxBranchingFactor == 2 ? maxBranchingFactor : random.ints(1, 2, maxBranchingFactor)
				.findFirst()
				.getAsInt();
		pltConfiguration.setK(k);

		AdaptivePLT plt = new AdaptivePLT(pltConfiguration);
		UUID learnerId = learnerRepository.create(plt, getId());
		pltCache.add(new PLTPropertiesForCache(learnerId));
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		pltCache.forEach(pltCacheEntry -> {
			UUID learnerId = pltCacheEntry.learnerId;
			AdaptivePLT learner = learnerRepository.read(learnerId, AdaptivePLT.class);
			learner.allocateClassifiers(data);

			pltCacheEntry.numberOfLabels = learner.m;
			learnerRepository.update(learnerId, learner);
		});
	}

	@Override
	public void train(DataManager data) {
		evaluate(data, true);
		int currentDataSetSize = 0;
		while (data.hasNext()) {

			Instance instance = data.getNextInstance();
			train(instance);
			currentDataSetSize++;
		}
		numberOfTrainingInstancesSeen += currentDataSetSize;

		tuneThreshold(data);

		evaluate(data, false);
	}

	private void train(Instance instance) {
		int epochs = minEpochs;

		for (PLTPropertiesForCache pltCacheEntry : pltCache) {

			UUID learnerId = pltCacheEntry.learnerId;
			AdaptivePLT learner = getAdaptivePLT(learnerId);

			learner.train(instance, epochs, true);
			double fm = learner.getFmeasureForInstance(instance, false, false);
			epochs = getNextEpochsFromFmeasure(fm);
			logger.info("Current fmeasure: " + fm + ", next epochs: " + epochs);

			// post processing
			// Collect and cache required data from plt
			pltCacheEntry.numberOfInstances = learner.getNumberOfTrainingInstancesSeen();
			pltCacheEntry.avgFmeasure = learner.getAverageFmeasure(false);
			pltCacheEntry.macroFmeasure = learner.getMacroFmeasure();
			pltCacheEntry.numberOfLabels = learner.m;

			// persist all changes happened during the training.
			learnerRepository.update(learnerId, learner);
		}

		Set<Integer> truePositive = new HashSet<Integer>(Ints.asList(instance.y));
		SetView<Integer> diff = Sets.difference(truePositive, labelsSeen);
		if (!diff.isEmpty()) {
			IAdaptiveTuner tuner = (IAdaptiveTuner) thresholdTuner;
			diff.forEach(label -> {
				tuner.accomodateNewLabel(label);
			});
		}
	}

	private AdaptivePLT getAdaptivePLT(UUID learnerId) {
		AdaptivePLT plt = learnerRepository.read(learnerId, AdaptivePLT.class);
		logger.info("Revived plt:" + learnerId);
		if (plt.fmeasureObserverAvailable) {
			plt.fmeasureObserver = fmeasureObserver;
			plt.addInstanceProcessedListener(fmeasureObserver);
		}
		return plt;
	}

	private int getNextEpochsFromFmeasure(double fm) {
		int epochs;
		if (fm == 0)
			fm = fZero;

		double alpha = 0.5 * log((1 - fm) / fm);
		epochs = toIntExact(round(minEpochs * exp(alpha)));

		PoissonDistribution pois = new PoissonDistribution(epochs);
		epochs = pois.sample();

		return epochs < minEpochs ? minEpochs : epochs;
	}

	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> predictions = new HashSet<Integer>();

		for (PLTPropertiesForCache pltCacheEntry : pltCache) {

			predictions.addAll(learnerRepository
					.read(pltCacheEntry.learnerId, AdaptivePLT.class)
					.getPositiveLabels(x));
		}

		return predictions;
	}

	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		List<int[]> predictions = getTopkLabelsFromEnsemble(x, k);

		if (predictions != null) {
			// Map predictions to Label to Set_Of_PLTs.
			ConcurrentHashMap<Integer, Set<PLTPropertiesForCache>> labelLearnerMap = new ConcurrentHashMap<Integer, Set<PLTPropertiesForCache>>();

			IntStream.range(0, predictions.size())
					.forEach(index -> Arrays
							.stream(predictions.get(index))
							.forEach(label -> {
								if (!labelLearnerMap.containsKey(label))
									labelLearnerMap.put(label, new HashSet<PLTPropertiesForCache>());
								labelLearnerMap.get(label)
										.add(pltCache.get(index));
							}));

			int ensembleSize = pltCache.size();

			// Assign score to each label
			Map<Integer, Double> labelScoreMap = labelLearnerMap.entrySet()
					.stream()
					.collect(Collectors.toMap(
							entry -> entry.getKey(),

							/* score = sum(avg. fmeasure of PLT predicting positive)/ensembleSize */
							entry -> (entry.getValue()
									.stream()
									.reduce(0.0,
											(sum, cachedPltDetails) -> sum += isToAggregateByMajorityVote ? 1
													: (preferMacroFmeasure
															? cachedPltDetails.macroFmeasure
															: cachedPltDetails.avgFmeasure),
											(sum1, sum2) -> sum1 + sum2))
									/ ensembleSize));

			// Sort the map by score (desc.) and return top k labels.
			return labelScoreMap.entrySet()
					.stream()
					.sorted(Entry.<Integer, Double>comparingByValue()
							.reversed())
					.limit(k)
					.mapToInt(entry -> entry.getKey())
					.toArray();
		}

		return new int[0];
	}

	private List<int[]> getTopkLabelsFromEnsemble(AVPair[] x, int k) {
		List<int[]> predictions = new ArrayList<int[]>();

		for (PLTPropertiesForCache pltCacheEntry : pltCache) {
			predictions.add(learnerRepository.read(pltCacheEntry.learnerId, AdaptivePLT.class)
					.getTopkLabels(x, k));
		}
		return predictions;
	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		return 0;
	}

	public double getMacroFmeasure() {
		return thresholdTuner.getMacroFmeasure();
	}
}
