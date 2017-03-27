package Learner;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

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

	private transient boolean isPredictionCacheActive = false;

	private PLTInitConfiguration pltConfiguration;
	private int maxPLTCacheSize = Integer.MIN_VALUE;

	/**
	 * Temp storage used in discardLearners, to avoid redefinition multiple time
	 */
	private transient List<PLTPropertiesForCache> scoredLearners;

	public PLTAdaptiveEnsemble(PLTAdaptiveEnsembleInitConfiguration configuration) {
		super(configuration);

		if (configuration.tunerInitOption == null || configuration.tunerInitOption.aSeed == null
				|| configuration.tunerInitOption.bSeed == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: invalid tuning option; aSeed and bSeed must be provided.");

		learnerRepository = configuration.learnerRepository;
		if (learnerRepository == null)
			throw new IllegalArgumentException(
					"Invalid init configuration: required learnerRepository object is not provided.");

		pltCache = new ArrayList<PLTPropertiesForCache>();
		pltCreatedListeners = new HashSet<IPLTCreatedListener>();
		pltDiscardedListeners = new HashSet<IPLTDiscardedListener>();
		labelsSeen = new TreeSet<Integer>();

		epsilon = configuration.getEpsilon();
		retainmentFraction = configuration.getRetainmentFraction();
		minTraingInstances = configuration.getMinTraingInstances();
		alpha = configuration.getAlpha();
		preferMacroFmeasure = configuration.isPreferMacroFmeasure();

		thresholdTuner = ThresholdTunerFactory.createThresholdTuner(0, ThresholdTuners.AdaptiveOfoFast,
				configuration.tunerInitOption);
		testTuner = ThresholdTunerFactory.createThresholdTuner(0, ThresholdTuners.AdaptiveOfoFast,
				configuration.tunerInitOption);
		testTopKTuner = ThresholdTunerFactory.createThresholdTuner(0, ThresholdTuners.AdaptiveOfoFast,
				configuration.tunerInitOption);

		penalizingStrategy = configuration.getPenalizingStrategy();
		ageFunction = configuration.getAgeFunction();

		c = configuration.getC();
		a = configuration.getA();

		pltConfiguration = configuration.individualPLTProperties;
		pltConfiguration.setToComputeFmeasureOnTopK(isToComputeFmeasureOnTopK);
		pltConfiguration.setDefaultK(defaultK);
	}

	@Override
	public void allocateClassifiers(DataManager data) {
	}

	private void addNewPLT(DataManager data) {
		if (fmeasureObserverAvailable)
			pltConfiguration.fmeasureObserver = fmeasureObserver;
		PLT plt = new PLT(pltConfiguration);
		plt.allocateClassifiers(data, labelsSeen);
		UUID learnerId = learnerRepository.create(plt, getId());
		pltCache.add(new PLTPropertiesForCache(learnerId, labelsSeen));
		onPLTCreated(plt);
	}

	@Override
	public void train(final DataManager data) {

		evaluate(data, true);

		double fmeasureOld = preferMacroFmeasure ? getMacroFmeasure()
				: getAverageFmeasure(false, isToComputeFmeasureOnTopK);
		double sumFmOld = preferMacroFmeasure ? -1 : fmeasureOld * getnTrain();

		int soFar = pltCache.size() > 0
				? pltCache.get(0).numberOfInstances
				: 0;

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		Instance instance;
		Set<Integer> truePositives;
		ImmutableSet<Integer> unseen;

		IAdaptiveTuner tuner = (IAdaptiveTuner) thresholdTuner;
		IAdaptiveTuner tstTuner = (IAdaptiveTuner) testTuner;
		IAdaptiveTuner tstTopkTuner = (IAdaptiveTuner) testTopKTuner;

		PLT plt = null;
		UUID learnerId;

		while (data.hasNext() == true) {

			instance = data.getNextInstance();

			truePositives = new HashSet<Integer>(Ints.asList(instance.y));
			unseen = Sets.difference(truePositives, labelsSeen)
					.immutableCopy();

			if (!unseen.isEmpty()) {
				labelsSeen.addAll(truePositives);
				addNewPLT(data);
				unseen.forEach(label -> {
					tuner.accomodateNewLabel(label);
					tstTuner.accomodateNewLabel(label);
					tstTopkTuner.accomodateNewLabel(label);
				});
			}

			for (PLTPropertiesForCache pltCacheEntry : pltCache) {

				learnerId = pltCacheEntry.learnerId;

				plt = getPLT(learnerId);
				plt.train(instance);

				// Collect and cache required data from plt
				pltCacheEntry.numberOfInstances = plt.getnTrain();
				if (preferMacroFmeasure)
					pltCacheEntry.macroFmeasure = plt.getMacroFmeasure();
				else
					pltCacheEntry.avgFmeasure = plt.getAverageFmeasure(false, isToComputeFmeasureOnTopK);

				// persist all changes happened during the training.
				learnerRepository.update(learnerId, plt);
			}
		}

		if (measureTime) {
			getStopwatch().stop();
			totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}

		if (maxPLTCacheSize < pltCache.size()) {
			maxPLTCacheSize = pltCache.size();
			logger.info("Max number of PLTs changed to " + maxPLTCacheSize);
		}

		nTrain += pltCache.get(0).numberOfInstances - soFar;

		activatePredictionCache();

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		double fmeasureNew = preferMacroFmeasure ? getTempMacroFMeasureOnData(data)
				: getTempFMeasureOnData(data, sumFmOld);
		logger.info("Old Fm: " + fmeasureOld + ", new Fm: " + fmeasureNew + ", epsilon: " + epsilon + ", diff: "
				+ Math.abs(fmeasureNew - fmeasureOld));
		if ((fmeasureOld - fmeasureNew) > epsilon) {
			discardLearners(sumFmOld, fmeasureOld, fmeasureNew, data);
		}

		if (measureTime) {
			getStopwatch().stop();
			totalTrainTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}

		tuneThreshold(data);

		evaluate(data, false);

		deactivatePredictionCache();
	}

	private PLT getPLT(UUID learnerId) {
		PLT plt = learnerRepository.read(learnerId, PLT.class);
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
	 */
	private void discardLearners(final double sumFmOld, final double fmeasureOld, double fmeasureNew,
			DataManager data) {

		scoredLearners = getPenalizedLearners().entrySet()
				.stream()
				.sorted(Entry.<PLTPropertiesForCache, Double>comparingByValue()
						.reversed())
				.map(entry -> entry.getKey())
				.collect(Collectors.toList());

		int minNumberOfPltsToRetain = (int) Math.ceil(pltCache.size() * retainmentFraction);
		PLTPropertiesForCache cachedPltDetails;
		while ((fmeasureOld - fmeasureNew) > epsilon && pltCache.size() > minNumberOfPltsToRetain
				&& scoredLearners.size() > 0) {

			cachedPltDetails = scoredLearners.remove(0);
			if (cachedPltDetails.numberOfInstances > minTraingInstances) {
				pltCache.remove(cachedPltDetails);
				onPLTDiscarded(cachedPltDetails);
				fmeasureNew = preferMacroFmeasure ? getTempMacroFMeasureOnData(data)
						: getTempFMeasureOnData(data, sumFmOld);

				logger.info("new fmeasure after discarding:" + fmeasureNew);
			}
		}

		scoredLearners.clear();
	}

	private Map<PLTPropertiesForCache, Double> getPenalizedLearners() {
		return pltCache.stream()
				.collect(Collectors.toMap(c -> c, c -> {

					double retVal = Double.MAX_VALUE;
					switch (penalizingStrategy) {
					case FmMinusRatioOfInstances:
						retVal = penalizeByFmMinusRatioOfInstances(c);
						break;
					case AgePlusLogOfInverseMacroFm:
						retVal = penalizeByAgePlusLogOfInverseMacroFm(c);
						break;
					default:
						throw new IllegalArgumentException("Unknown penalizing strategy: " + penalizingStrategy);
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
	private double penalizeByFmMinusRatioOfInstances(PLTPropertiesForCache cachedPltDetails) {
		return 1 - (alpha * (preferMacroFmeasure ? cachedPltDetails.macroFmeasure : cachedPltDetails.avgFmeasure)
				- (1 - alpha) * ((double) cachedPltDetails.numberOfInstances / getnTrain()));
	}

	private double penalizeByAgePlusLogOfInverseMacroFm(PLTPropertiesForCache cachedPltDetails) {
		return alpha * pow(c * getAgeOfPlt(cachedPltDetails), a)
				+ (1 - alpha) * pow(
						log(1.0 / (preferMacroFmeasure ? cachedPltDetails.macroFmeasure
								: cachedPltDetails.avgFmeasure)),
						a);
	}

	private double getAgeOfPlt(PLTPropertiesForCache cachedPltDetails) {
		double retVal = 0;
		switch (ageFunction) {

		case NumberOfLabelsBased:
			retVal = exp(-(double) cachedPltDetails.numberOfLabels / (double) labelsSeen.size());
			break;

		case NumberTrainingInstancesBased:
			retVal = (double) cachedPltDetails.numberOfInstances / (double) nTrain;
			break;

		default:
			throw new IllegalArgumentException("Unknown age function: " + ageFunction);
		}

		return retVal;
	}

	private double getTempFMeasureOnData(DataManager data, double sumFmOld) {

		while (data.hasNext() == true) {
			sumFmOld += getFmeasureForInstance(data.getNextInstance());
		}
		data.reset();
		return sumFmOld / getnTrain();
	}

	private double getTempMacroFMeasureOnData(DataManager data) {
		return thresholdTuner.getTempMacroFmeasure(createTuningData(data));
	}

	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> predictions = new HashSet<Integer>();

		pltCache.forEach(pltCacheEntry -> {
			if (isPredictionCacheActive && pltCacheEntry.tempPredictions.containsKey(x))
				predictions.addAll(pltCacheEntry.tempPredictions.get(x));
			else {
				predictions.addAll(learnerRepository
						.read(pltCacheEntry.learnerId, PLT.class)
						.getPositiveLabels(x));
				if (isPredictionCacheActive)
					pltCacheEntry.tempPredictions.put(x, predictions);
			}
		});

		return predictions;
	}

	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		List<int[]> predictions = getTopkLabelsFromEnsemble(x, k);

		if (predictions != null && !predictions.isEmpty()) {
			// Map predictions to Label to Set_Of_PLTs.
			ConcurrentHashMap<Integer, Set<PLTPropertiesForCache>> labelLearnerMap = new ConcurrentHashMap<Integer, Set<PLTPropertiesForCache>>();

			for (int index = 0; index < predictions.size(); index++)
				for (int label : predictions.get(index)) {
					if (!labelLearnerMap.containsKey(label))
						labelLearnerMap.put(label, new HashSet<PLTPropertiesForCache>());
					labelLearnerMap.get(label)
							.add(pltCache.get(index));
				}

			// Assign score to each label
			Map<Integer, Double> labelScoreMap = labelLearnerMap.entrySet()
					.stream()
					.collect(Collectors.toMap(
							entry -> entry.getKey(),

							/* score = sum(avg. fmeasure of PLT predicting positive)/numberOfPLTsHavingThisLabel */
							entry -> {
								Integer label = entry.getKey();
								long numberOfPltsHavingLabel = pltCache.stream()
										.filter(cachedPltDetails -> cachedPltDetails.labels.contains(label))
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

		return new int[0];
	}

	private List<int[]> getTopkLabelsFromEnsemble(AVPair[] x, int k) {
		List<int[]> predictions = new ArrayList<int[]>();

		pltCache.forEach(pltCacheEntry -> {
			if (isPredictionCacheActive && pltCacheEntry.tempTopkPredictions.containsKey(x))
				predictions.add(pltCacheEntry.tempTopkPredictions.get(x));
			else {
				int[] topkLabels = learnerRepository.read(pltCacheEntry.learnerId, PLT.class)
						.getTopkLabels(x, k);
				predictions.add(topkLabels);
				if (isPredictionCacheActive)
					pltCacheEntry.tempTopkPredictions.put(x, topkLabels);
			}
		});
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

		pltCreatedListeners.forEach(listener -> listener.onPLTCreated(this, args));
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

		pltDiscardedListeners.forEach(listener -> listener.onPLTDiscarded(this, args));
	}

	@Override
	protected void tuneThreshold(DataManager data) {
		thresholdTuner.getTunedThresholdsSparse(createTuningData(data));
	}

	private void deactivatePredictionCache() {
		isPredictionCacheActive = false;
		pltCache.forEach(plt -> plt.clearAllPredictions());
	}

	private void activatePredictionCache() {
		isPredictionCacheActive = true;
		pltCache.forEach(plt -> plt.clearAllPredictions());
	}
}