package Learner;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;

import Data.AVPair;
import Data.Instance;
import IO.DataManager;
import event.args.PLTCreationEventArgs;
import event.args.PLTDiscardedEventArgs;
import event.listeners.IPLTCreatedListener;
import event.listeners.IPLTDiscardedListener;
import interfaces.ILearnerRepository;
import threshold.IAdaptiveTuner;
import threshold.ThresholdTunerFactory;
import threshold.ThresholdTuners;
import util.LearnerInitConfiguration;
import util.PLTAdaptiveEnsembleAgeFunctions;
import util.PLTAdaptiveEnsembleInitConfiguration;
import util.PLTAdaptiveEnsemblePenalizingStrategies;
import util.PLTInitConfiguration;
import util.PLTPropertiesForCache;

public class PLTAdaptiveEnsemble extends AbstractLearner {
	private static final long serialVersionUID = 7193120904682573610L;

	private static Logger logger = LoggerFactory.getLogger(PLTAdaptiveEnsemble.class);

	transient private ILearnerRepository learnerRepository;
	private List<PLTPropertiesForCache> pltCache;

	transient private Set<IPLTCreatedListener> pltCreatedListeners;
	transient private Set<IPLTDiscardedListener> pltDiscardedListeners;

	/**
	 * Slack variable for fmeasure comparison.
	 */
	private final double epsilon;
	/**
	 * Fraction of learners to retain in the discarding phase.
	 */
	private final double retainmentFraction;
	/**
	 * Minimum number of training instances an individual PLT should be trained
	 * on, before discarding.
	 */
	private final int minTraingInstances;
	/**
	 * Relative weight ([0, 1]) used to penalize the learner during discarding.
	 */
	private final double alpha;
	/**
	 * Prefer macro fmeasure for discarding learners and aggregating result.
	 */
	private boolean preferMacroFmeasure;

	private SortedSet<Integer> labelsSeen;
	private PLTAdaptiveEnsemblePenalizingStrategies penalizingStrategy;
	private PLTAdaptiveEnsembleAgeFunctions ageFunction;

	private int c;
	private int a;

	private PLTInitConfiguration pltConfiguration;

	public PLTAdaptiveEnsemble(LearnerInitConfiguration configuration) throws Exception {
		super(configuration);

		PLTAdaptiveEnsembleInitConfiguration ensembleConfiguration = configuration instanceof PLTAdaptiveEnsembleInitConfiguration
				? (PLTAdaptiveEnsembleInitConfiguration) configuration : null;
		if (ensembleConfiguration == null)
			throw new Exception("Invalid init configuration");

		if (ensembleConfiguration.tunerInitOption == null || ensembleConfiguration.tunerInitOption.aSeed == null
				|| ensembleConfiguration.tunerInitOption.bSeed == null)
			throw new Exception("Invalid tuning option; aSeed and bSeed must be provided.");

		learnerRepository = ensembleConfiguration.learnerRepository;
		if (learnerRepository == null)
			throw new Exception(
					"Invalid initialization parameters. A required learnerRepository object is not provided.");

		pltCache = new ArrayList<PLTPropertiesForCache>();
		pltCreatedListeners = new HashSet<IPLTCreatedListener>();
		pltDiscardedListeners = new HashSet<IPLTDiscardedListener>();
		labelsSeen = new TreeSet<Integer>();

		epsilon = ensembleConfiguration.getEpsilon();
		retainmentFraction = ensembleConfiguration.getRetainmentFraction();
		minTraingInstances = ensembleConfiguration.getMinTraingInstances();
		alpha = ensembleConfiguration.getAlpha();
		preferMacroFmeasure = ensembleConfiguration.isPreferMacroFmeasure();

		thresholdTuner = ThresholdTunerFactory.createThresholdTuner(1, ThresholdTuners.AdaptiveOfoFast,
				ensembleConfiguration.tunerInitOption);

		penalizingStrategy = ensembleConfiguration.getPenalizingStrategy();
		ageFunction = ensembleConfiguration.getAgeFunction();

		c = ensembleConfiguration.getC();
		a = ensembleConfiguration.getA();

		pltConfiguration = ensembleConfiguration.individualPLTProperties;
		pltConfiguration.setToComputeFmeasureOnTopK(isToComputeFmeasureOnTopK);
		pltConfiguration.setDefaultK(defaultK);
		if (fmeasureObserverAvailable)
			pltConfiguration.fmeasureObserver = fmeasureObserver;
	}

	@Override
	public void allocateClassifiers(DataManager data) {
	}

	private void addNewPLT(DataManager data) throws Exception {
		PLT plt = new PLT(pltConfiguration);
		plt.allocateClassifiers(data, labelsSeen);
		UUID learnerId = learnerRepository.create(plt, getId());
		pltCache.add(new PLTPropertiesForCache(learnerId, plt.m));
		onPLTCreated(plt);
	}

	@Override
	public void train(final DataManager data) throws Exception {

		double fmeasureOld = preferMacroFmeasure ? getMacroFmeasure() : getAverageFmeasure(false);
		double sumFmOld = preferMacroFmeasure ? -1 : fmeasureOld * getNumberOfTrainingInstancesSeen();

		int soFar = pltCache.size() > 0
				? pltCache.get(0).numberOfInstances
				: 0;

		while (data.hasNext() == true) {

			Instance instance = data.getNextInstance();

			Set<Integer> truePositives = new HashSet<Integer>(Ints.asList(instance.y));
			ImmutableSet<Integer> unseen = Sets.difference(truePositives, labelsSeen)
					.immutableCopy();

			if (!unseen.isEmpty()) {
				labelsSeen.addAll(truePositives);
				addNewPLT(data);

				IAdaptiveTuner tuner = (IAdaptiveTuner) thresholdTuner;
				unseen.forEach(label -> {
					tuner.accomodateNewLabel(label);
				});
			}

			for (PLTPropertiesForCache pltCacheEntry : pltCache) {

				UUID learnerId = pltCacheEntry.learnerId;

				PLT plt = getPLT(learnerId);
				logger.info("Training " + learnerId);
				plt.train(instance);

				// Collect and cache required data from plt
				pltCacheEntry.numberOfInstances = plt.getNumberOfTrainingInstancesSeen();
				if (preferMacroFmeasure)
					pltCacheEntry.macroFmeasure = plt.getMacroFmeasure();
				else
					pltCacheEntry.avgFmeasure = plt.getAverageFmeasure(false);

				// persist all changes happened during the training.
				learnerRepository.update(learnerId, plt);
			}
		}

		int numberOfTrainingInstancesInThisSession = pltCache.get(0).numberOfInstances - soFar;
		numberOfTrainingInstancesSeen += numberOfTrainingInstancesInThisSession;

		double fmeasureNew = preferMacroFmeasure ? getTempMacroFMeasureOnData(data)
				: getTempFMeasureOnData(data, sumFmOld);
		logger.info("Old Fm: " + fmeasureOld + ", new Fm: " + fmeasureNew + ", epsilon: " + epsilon + ", diff: "
				+ Math.abs(fmeasureNew - fmeasureOld));
		if ((fmeasureOld - fmeasureNew) > epsilon) {
			discardLearners(sumFmOld, fmeasureOld, fmeasureNew, data);
		}

		tuneThreshold(data);

		evaluate(data, false);
	}

	private PLT getPLT(UUID learnerId) {
		PLT plt = learnerRepository.read(learnerId, PLT.class);
		logger.info("Revived plt:" + learnerId);
		if (plt.fmeasureObserverAvailable) {
			plt.fmeasureObserver = fmeasureObserver;
			plt.addInstanceProcessedListener(fmeasureObserver);
		}
		return plt;
	}

	/**
	 * Orders the learners by score {@code PLT#scoringStrategy}, discards low
	 * scoring learners until termination criteria is met. <br/>
	 * <br/>
	 * Termination criteria:
	 * {@code (fmeasureNew - fmeasureOld > epsilon) OR (plts.size() <= minimum number of PLTs to retain) }
	 * 
	 * @param sumFmOld
	 *            Sum of old fmeasures (approximated with
	 *            {@code numberOfTrainingInstancesSeenPreviously * averageFmeasure}).
	 * @param fmeasureOld
	 * @param fmeasureNew
	 * @param data
	 * @throws Exception
	 */
	private void discardLearners(final double sumFmOld, final double fmeasureOld, double fmeasureNew,
			DataManager data) throws Exception {

		List<PLTPropertiesForCache> scoredLearnerIds = getPenalizedLearners().entrySet()
				.stream()
				.sorted(Entry.<PLTPropertiesForCache, Double>comparingByValue()
						.reversed())
				.map(entry -> entry.getKey())
				.collect(Collectors.toList());

		int minNumberOfPltsToRetain = (int) Math.ceil(pltCache.size() * retainmentFraction);

		while ((fmeasureOld - fmeasureNew) > epsilon && pltCache.size() > minNumberOfPltsToRetain
				&& scoredLearnerIds.size() > 0) {

			PLTPropertiesForCache cachedPltDetails = scoredLearnerIds.remove(0);
			if (cachedPltDetails.numberOfInstances > minTraingInstances) {
				pltCache.remove(cachedPltDetails);
				onPLTDiscarded(cachedPltDetails);
				fmeasureNew = preferMacroFmeasure ? getTempMacroFMeasureOnData(data)
						: getTempFMeasureOnData(data, sumFmOld);

				logger.info("new fmeasure after discarding:" + fmeasureNew);
			}
		}
	}

	private Map<PLTPropertiesForCache, Double> getPenalizedLearners() {
		return pltCache.stream()
				.collect(Collectors.toMap(c -> c, c -> {

					double retVal = Double.MAX_VALUE;
					try {
						switch (penalizingStrategy) {
						case MacroFmMinusRatioOfInstances:
							retVal = penalizeByMacroFmMinusRatioOfInstances(c);
							break;
						case AgePlusLogOfInverseMacroFm:
							retVal = penalizeByAgePlusLogOfInverseMacroFm(c);
							break;
						default:
							throw new Exception("Unknown penalizing strategy: " + penalizingStrategy);
						}
					} catch (Exception e) {
						logger.error(e.getMessage(), e);
						System.exit(-1);
					}
					return retVal;
				}));
	}

	/**
	 * 
	 * @param cachedPltDetails
	 * @return The score of {@code plt} as
	 *         {@code 1 - (avgFmeasureOfPlt - (numberOfTrainingInstancesSeenByPlt/TotalNumberOfTrainingInstancesSeenSoFar))}
	 */
	private double penalizeByMacroFmMinusRatioOfInstances(PLTPropertiesForCache cachedPltDetails) {
		return 1 - (alpha * (preferMacroFmeasure ? cachedPltDetails.macroFmeasure : cachedPltDetails.avgFmeasure)
				- (1 - alpha) * ((double) cachedPltDetails.numberOfInstances / getNumberOfTrainingInstancesSeen()));
	}

	private double penalizeByAgePlusLogOfInverseMacroFm(PLTPropertiesForCache cachedPltDetails) throws Exception {
		return alpha * pow(c * getAgeOfPlt(cachedPltDetails), a)
				+ (1 - alpha) * pow(log(1 / cachedPltDetails.macroFmeasure), a);
	}

	private double getAgeOfPlt(PLTPropertiesForCache cachedPltDetails) throws Exception {
		double retVal = 0;
		switch (ageFunction) {

		case NumberOfLabelsBased:
			retVal = exp(-(double) cachedPltDetails.numberOfLabels / (double) labelsSeen.size());
			break;

		case NumberTrainingInstancesBased:
			retVal = (double) cachedPltDetails.numberOfInstances / (double) numberOfTrainingInstancesSeen;
			break;

		default:
			throw new Exception("Unknown age function: " + ageFunction);
		}

		return retVal;
	}

	private double getTempFMeasureOnData(DataManager data, double sumFmOld) {

		while (data.hasNext() == true) {
			sumFmOld += getFmeasureForInstance(data.getNextInstance());
		}
		data.reset();
		return sumFmOld / getNumberOfTrainingInstancesSeen();
	}

	private double getTempMacroFMeasureOnData(DataManager data) throws Exception {
		return thresholdTuner.getTempMacroFmeasure(createTuningData(data));
	}

	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> predictions = new HashSet<Integer>();

		for (PLTPropertiesForCache pltCacheEntry : pltCache) {

			logger.info("Getting predictions from " + pltCacheEntry.learnerId);
			predictions.addAll(learnerRepository
					.read(pltCacheEntry.learnerId, PLT.class)
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

			// Assign score to each label
			Map<Integer, Double> labelScoreMap = labelLearnerMap.entrySet()
					.stream()
					.collect(Collectors.toMap(
							entry -> entry.getKey(),

							/* score = sum(avg. fmeasure of PLT predicting positive)/numberOfPLTsHavingThisLabel */
							entry -> {
								Integer label = entry.getKey();
								long numberOfPltsHavingLabel = pltCache.stream()
										.filter(cachedPltDetails -> cachedPltDetails.numberOfLabels > label)
										.count();
								return (entry.getValue()
										.stream()
										.reduce(0.0,
												(sum, cachedPltDetails) -> sum += (preferMacroFmeasure
														? cachedPltDetails.macroFmeasure
														: cachedPltDetails.avgFmeasure),
												(sum1, sum2) -> sum1 + sum2))
										/ numberOfPltsHavingLabel;
							}));

			// Sort the map by score (desc.) and return top k labels.
			return labelScoreMap.entrySet()
					.stream()
					.sorted(Entry.<Integer, Double>comparingByValue()
							.reversed())
					.limit(k)
					.mapToInt(entry -> entry.getKey())
					.toArray();
		}

		return null;
	}

	private List<int[]> getTopkLabelsFromEnsemble(AVPair[] x, int k) {
		List<int[]> predictions = new ArrayList<int[]>();

		for (PLTPropertiesForCache pltCacheEntry : pltCache) {
			predictions.add(learnerRepository.read(pltCacheEntry.learnerId, PLT.class)
					.getTopkLabels(x, k));
		}
		return predictions;
	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		// TODO Auto-generated method stub
		return 0;
	}

	public void addPLTCreatedListener(IPLTCreatedListener listener) {
		pltCreatedListeners.add(listener);
	}

	public void removePLTCreatedListener(IPLTCreatedListener listener) {
		pltCreatedListeners.remove(listener);
	}

	private void onPLTCreated(PLT plt) {
		PLTCreationEventArgs args = new PLTCreationEventArgs();
		args.plt = plt;

		pltCreatedListeners.stream()
				.forEach(listener -> listener.onPLTCreated(this, args));
	}

	public void addPLTDiscardedListener(IPLTDiscardedListener listener) {
		pltDiscardedListeners.add(listener);
	}

	public void removePLTDiscardedListener(IPLTDiscardedListener listener) {
		pltDiscardedListeners.remove(listener);
	}

	private void onPLTDiscarded(PLTPropertiesForCache plt) {
		PLTDiscardedEventArgs args = new PLTDiscardedEventArgs();
		args.pltId = plt.learnerId;

		pltDiscardedListeners.stream()
				.forEach(listener -> listener.onPLTDiscarded(this, args));
	}

	public double getMacroFmeasure() {
		return thresholdTuner.getMacroFmeasure();
	}

	@Override
	protected void tuneThreshold(DataManager data) {
		try {
			thresholdTuner.getTunedThresholdsSparse(createTuningData(data));
		} catch (Exception e) {
			logger.error("Error during tuning the threshlds.", e);
			System.exit(-1);
		}
	}
}