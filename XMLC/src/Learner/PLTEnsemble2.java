package Learner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.Instance;
import IO.DataManager;
import event.args.PLTCreationEventArgs;
import event.args.PLTDiscardedEventArgs;
import event.listeners.IPLTCreatedListener;
import event.listeners.IPLTDiscardedListener;
import interfaces.ILearnerRepository;
import util.Constants;
import util.PLTPropertiesForCache;
import util.Constants.LearnerInitProperties;

public class PLTEnsemble2 extends AbstractLearner {
	private static final long serialVersionUID = 7193120904682573610L;

	private static Logger logger = LoggerFactory.getLogger(PLTEnsemble2.class);

	transient private ILearnerRepository learnerRepository;
	private int currentNumberOfLabels = 0;
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
	 * Weight ([0, 1]) to be applied on fmeasure, while scoring the learner
	 * before discarding
	 */
	private final double alpha;

	public PLTEnsemble2(Properties properties) throws Exception {
		super(properties);

		learnerRepository = (ILearnerRepository) properties.get(LearnerInitProperties.learnerRepository);
		if (learnerRepository == null)
			throw new Exception(
					"Invalid initialization parameters. A required learnerRepository object is not provided.");

		pltCache = new ArrayList<PLTPropertiesForCache>();
		pltCreatedListeners = new HashSet<IPLTCreatedListener>();
		pltDiscardedListeners = new HashSet<IPLTDiscardedListener>();

		epsilon = Double.parseDouble(
				properties.getProperty(LearnerInitProperties.pltEnsembleEpsilon,
						Double.toString(Constants.PLTEnsembleDefaultValues.epsilon)));
		retainmentFraction = Double
				.parseDouble(properties.getProperty(LearnerInitProperties.pltEnsembleRetainmentFraction,
						Double.toString(Constants.PLTEnsembleDefaultValues.retainmentFraction)));
		minTraingInstances = Integer.parseInt(properties.getProperty(LearnerInitProperties.minTraingInstances,
				Integer.toString(Constants.PLTEnsembleDefaultValues.minTraingInstances)));

		alpha = Double.parseDouble(
				properties.getProperty(LearnerInitProperties.pltEnsembleAlpha,
						Double.toString(Constants.PLTEnsembleDefaultValues.alpha)));
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		if (pltCache.isEmpty()) {
			addNewPLT(data);
		}
	}

	private void addNewPLT(DataManager data) {
		Properties pltProperties = (Properties) properties.get(LearnerInitProperties.individualPLTProperties);
		pltProperties.put(LearnerInitProperties.isToComputeFmeasureOnTopK, isToComputeFmeasureOnTopK);
		pltProperties.put(LearnerInitProperties.defaultK, defaultK);
		if (fmeasureObserverAvailable)
			pltProperties.put(LearnerInitProperties.fmeasureObserver, fmeasureObserver);

		PLT plt = new PLT(pltProperties);
		plt.allocateClassifiers(data);
		UUID learnerId = learnerRepository.create(plt, getId());
		pltCache.add(new PLTPropertiesForCache(learnerId, plt.m));
		currentNumberOfLabels = data.getNumberOfLabels();
		onPLTCreated(plt);
	}

	@Override
	public void train(final DataManager data) {

		double fmeasureOld = getAverageFmeasure(false);
		double sumFmOld = fmeasureOld * getNumberOfTrainingInstancesSeen();

		int soFar = pltCache.size() > 0
				? pltCache.get(0).numberOfInstances
				: 0;

		if (data.getNumberOfLabels() > currentNumberOfLabels)
			addNewPLT(data);

		while (data.hasNext() == true) {
			Instance instance = data.getNextInstance();

			for (PLTPropertiesForCache pltCacheEntry : pltCache) {

				UUID learnerId = pltCacheEntry.learnerId;

				PLT plt = getPLT(learnerId);
				logger.info("Training " + learnerId);
				plt.train(instance);

				// Collect and cache required data from plt
				pltCacheEntry.numberOfInstances = plt.getNumberOfTrainingInstancesSeen();
				pltCacheEntry.avgFmeasure = plt.getAverageFmeasure(false);

				// persist all changes happened during the training.
				learnerRepository.update(learnerId, plt);
			}
		}

		int numberOfTrainingInstancesInThisSession = pltCache.get(0).numberOfInstances - soFar;
		numberOfTrainingInstancesSeen += numberOfTrainingInstancesInThisSession;

		double fmeasureNew = getTempFMeasureOnData(data, sumFmOld);
		logger.info("Old Fm: " + fmeasureOld + ", new Fm: " + fmeasureNew + ", epsilon: " + epsilon + ", diff: "
				+ Math.abs(fmeasureNew - fmeasureOld));
		if (fmeasureNew < fmeasureOld && Math.abs(fmeasureNew - fmeasureOld) > epsilon) {
			discardLearners(sumFmOld, fmeasureOld, fmeasureNew, data);
		}

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
	 */
	private void discardLearners(final double sumFmOld, final double fmeasureOld, double fmeasureNew,
			DataManager data) {

		List<PLTPropertiesForCache> scoredLearnerIds = getScoredLearners().entrySet()
				.stream()
				.sorted(Entry.<PLTPropertiesForCache, Double>comparingByValue())
				.map(entry -> entry.getKey())
				.collect(Collectors.toList());

		int minNumberOfPltsToRetain = (int) Math.ceil(pltCache.size() * retainmentFraction);

		while (/*Math.abs(*/fmeasureOld - fmeasureNew/*)*/ > epsilon && pltCache.size() > minNumberOfPltsToRetain
				&& scoredLearnerIds.size() > 0) {

			PLTPropertiesForCache cachedPltDetails = scoredLearnerIds.remove(0);
			if (cachedPltDetails.numberOfInstances > minTraingInstances) {
				pltCache.remove(cachedPltDetails);
				onPLTDiscarded(cachedPltDetails);
				fmeasureNew = getTempFMeasureOnData(data, sumFmOld);
				logger.info("new fmeasure after discarding:" + fmeasureNew);
			}
		}
	}

	private Map<PLTPropertiesForCache, Double> getScoredLearners() {
		return pltCache.stream()
				.collect(Collectors.toMap(c -> c, c -> scoringStrategy(c)));
	}

	/**
	 * 
	 * @param cachedPltDetails
	 * @return The score of {@code plt} as
	 *         {@code avgFmeasureOfPlt - (numberOfTrainingInstancesSeenByPlt/TotalNumberOfTrainingInstancesSeenSoFar)}
	 */
	private double scoringStrategy(PLTPropertiesForCache cachedPltDetails) {
		return alpha * cachedPltDetails.avgFmeasure
				- (1 - alpha) * ((double) cachedPltDetails.numberOfInstances / getNumberOfTrainingInstancesSeen());
	}

	private double getTempFMeasureOnData(DataManager data, double sumFmOld) {

		while (data.hasNext() == true) {
			sumFmOld += getFmeasureForInstance(data.getNextInstance());
		}
		data.reset();
		return sumFmOld / getNumberOfTrainingInstancesSeen();
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

		// ExecutorService executor = Executors.newWorkStealingPool();
		// List<Callable<HashSet<Integer>>> tasks = new
		// ArrayList<Callable<HashSet<Integer>>>();
		//
		// for (PLTPropertiesForCache pltCacheEntry : pltCache) {
		// tasks.add(() -> {
		// logger.info("Getting predictions from " + pltCacheEntry.learnerId);
		// return learnerRepository.read(pltCacheEntry.learnerId, PLT.class)
		// .getPositiveLabels(x);
		// });
		// }
		//
		// HashSet<Integer> predictions = null;
		//
		// try {
		// predictions = executor.invokeAll(tasks)
		// .stream()
		// .flatMap(future -> {
		// try {
		// return future.get()
		// .stream();
		// } catch (InterruptedException | ExecutionException e) {
		// throw new IllegalStateException(e);
		// }
		// })
		// .collect(Collectors.toCollection(HashSet::new));
		//
		// executor.shutdown();
		// executor.awaitTermination(Integer.MAX_VALUE, TimeUnit.MILLISECONDS);
		// } catch (InterruptedException e) {
		// logger.warn(
		// "Threds interrupted. Prediction took more than " + Integer.MAX_VALUE
		// + " " + TimeUnit.MILLISECONDS
		// + " to finish.");
		// }

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
												(sum, cachedPltDetails) -> sum += cachedPltDetails.avgFmeasure,
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
}