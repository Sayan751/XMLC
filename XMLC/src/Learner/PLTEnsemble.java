package Learner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import IO.DataManager;
import event.args.PLTCreationEventArgs;
import event.args.PLTDiscardedEventArgs;
import event.listeners.IPLTCreatedListener;
import event.listeners.IPLTDiscardedListener;
import util.Constants;
import util.Constants.LearnerInitProperties;

public class PLTEnsemble extends AbstractLearner {
	private static final long serialVersionUID = 7193120904682573610L;

	private static Logger logger = LoggerFactory.getLogger(PLTEnsemble.class);
	private List<PLT> plts;
	private int currentNumberOfLabels = 0;
	private Map<PLT, Double> averageFmeasureCache;

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

	public PLTEnsemble(Properties properties) {
		super(properties);

		plts = new ArrayList<PLT>();
		averageFmeasureCache = new HashMap<PLT, Double>();
		pltCreatedListeners = new HashSet<IPLTCreatedListener>();
		pltDiscardedListeners = new HashSet<IPLTDiscardedListener>();

		epsilon = Double.parseDouble(
				properties.getProperty(LearnerInitProperties.pltEnsembleEpsilon,
						Double.toString(Constants.PLTEnsembleDefaultValues.epsilon)));
		retainmentFraction = Double
				.parseDouble(properties.getProperty(LearnerInitProperties.pltEnsembleRetainmentFraction,
						Double.toString(Constants.PLTEnsembleDefaultValues.retainmentFraction)));
		minTraingInstances = Integer.parseInt(properties.getProperty(LearnerInitProperties.pltEnsembleEpsilon,
				Integer.toString(Constants.PLTEnsembleDefaultValues.minTraingInstances)));

	}

	@Override
	public void allocateClassifiers(DataManager data) {
		if (plts.isEmpty()) {
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
		plts.add(plt);
		currentNumberOfLabels = data.getNumberOfLabels();
		onPLTCreated(plt);
	}

	@Override
	public void train(final DataManager data) {

		double fmeasureOld = getAverageFmeasure(false);
		int soFar = plts.size() > 0
				? plts.get(0)
						.getNumberOfTrainingInstancesSeen()
				: 0;

		if (data.getNumberOfLabels() > currentNumberOfLabels)
			addNewPLT(data);

		// ExecutorService executor = Executors.newWorkStealingPool();
		// for (PLT plt : plts) {
		// executor.submit(() -> {
		// logger.info("Training " + plt);
		// plt.train(data);
		// });
		// }
		//
		// executor.shutdown();
		// try {
		// executor.awaitTermination(Integer.MAX_VALUE, TimeUnit.MILLISECONDS);
		// } catch (InterruptedException e) {
		// logger.warn("Threds interrupted. Tasks took more than " +
		// Integer.MAX_VALUE + " " + TimeUnit.MILLISECONDS
		// + " to finish.");
		// }

		for (PLT plt : plts) {
			plt.train(data);
		}

		int numberOfTrainingInstancesInThisSession = plts.get(0)
				.getNumberOfTrainingInstancesSeen() - soFar;

		averageFmeasureCache.clear();
		averageFmeasureCache = plts.stream()
				.collect(Collectors.toMap(
						plt -> plt,
						plt -> plt.getAverageFmeasure(false)));

		double fmeasureNew = getTempFMeasureOnData(data, fmeasureOld);
		if (fmeasureNew < fmeasureOld && Math.abs(fmeasureNew - fmeasureOld) > epsilon) {
			discardLearners(fmeasureOld, fmeasureNew, data);
		}

		evaluate(data, false);
		numberOfTrainingInstancesSeen += numberOfTrainingInstancesInThisSession;
	}

	/**
	 * Orders the learners by score {@code PLT#scoringStrategy}, discards low
	 * scoring learners until termination criteria is met. <br/>
	 * <br/>
	 * Termination criteria:
	 * {@code (|fmeasureOld - fmeasureNew| <= epsilon) OR (plts.size() <= minimum number of PLTs to retain) }
	 * 
	 * @param fmeasureOld
	 * @param fmeasureNew
	 * @param data
	 */
	private void discardLearners(final double fmeasureOld, double fmeasureNew, DataManager data) {
		List<PLT> scoredLearners = getScoredLearners().entrySet()
				.stream()
				.sorted(Entry.<PLT, Double>comparingByValue())
				.map(entry -> entry.getKey())
				.collect(Collectors.toList());

		int minNumberOfPltsToRetain = (int) Math.ceil(plts.size() * retainmentFraction);

		while (Math.abs(fmeasureNew - fmeasureOld) > epsilon && plts.size() > minNumberOfPltsToRetain
				&& scoredLearners.size() > 0) {

			PLT plt = scoredLearners.remove(0);
			if (plt.numberOfTrainingInstancesSeen > minTraingInstances) {
				plts.remove(plt);
				averageFmeasureCache.remove(plt);
				onPLTDiscarded(plt);
			}

			fmeasureNew = getTempFMeasureOnData(data, fmeasureOld);
		}
	}

	private Map<PLT, Double> getScoredLearners() {
		return plts.stream()
				.collect(Collectors.toMap(plt -> plt, plt -> scoringStrategy(plt)));
	}

	/**
	 * 
	 * @param plt
	 * @return The score of {@code plt} as
	 *         {@code avgFmeasureOfPlt - (numberOfTrainingInstancesSeenByPlt/TotalNumberOfTrainingInstancesSeenSoFar)}
	 */
	private double scoringStrategy(PLT plt) {
		return averageFmeasureCache.get(plt)
				- ((double) plt.getNumberOfTrainingInstancesSeen() / getNumberOfTrainingInstancesSeen());
	}

	private double getTempFMeasureOnData(DataManager data, double fmeasureOld) {
		List<Double> resultantFmeasures = new ArrayList<Double>();
		while (data.hasNext() == true) {
			resultantFmeasures.add(getFmeasureForInstance(data.getNextInstance()));
		}
		data.reset();
		return (resultantFmeasures.stream()
				.mapToDouble(d -> d)
				.sum() + fmeasureOld) / (resultantFmeasures.size() + numberOfTrainingInstancesSeen);
	}

	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		ExecutorService executor = Executors.newWorkStealingPool();
		List<Callable<HashSet<Integer>>> tasks = new ArrayList<Callable<HashSet<Integer>>>();

		for (PLT plt : plts) {
			tasks.add(() -> {
				logger.info("Getting predictions from " + plt);
				return plt.getPositiveLabels(x);
			});
		}

		HashSet<Integer> predictions = null;

		try {
			predictions = executor.invokeAll(tasks)
					.stream()
					.flatMap(future -> {
						try {
							return future.get()
									.stream();
						} catch (InterruptedException | ExecutionException e) {
							throw new IllegalStateException(e);
						}
					})
					.collect(Collectors.toCollection(HashSet::new));

			executor.shutdown();
			executor.awaitTermination(Integer.MAX_VALUE, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			logger.warn(
					"Threds interrupted. Prediction took more than " + Integer.MAX_VALUE + " " + TimeUnit.MILLISECONDS
							+ " to finish.");
		}

		return predictions;
	}

	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		List<int[]> predictions = getTopkLabelsFromEnsemble(x, k);

		if (predictions != null) {
			// Map predictions to Label to Set_Of_PLTs.
			ConcurrentHashMap<Integer, Set<PLT>> labelLearnerMap = new ConcurrentHashMap<Integer, Set<PLT>>();

			IntStream.range(0, predictions.size())
					// .parallel()
					.forEach(index -> Arrays
							.stream(predictions.get(index))
							.forEach(label -> {
								if (!labelLearnerMap.containsKey(label))
									labelLearnerMap.put(label, new HashSet<PLT>());
								labelLearnerMap.get(label)
										.add(plts.get(index));
							}));

			// Assign score to each label
			Map<Integer, Double> labelScoreMap = labelLearnerMap.entrySet()
					.stream()
					.collect(Collectors.toMap(
							entry -> entry.getKey(),

							/* score = sum(avg. fmeasure of PLT predicting positive)/numberOfPLTsHavingThisLabel */
							entry -> {
								Integer label = entry.getKey();
								long numberOfPltsHavingLabel = plts.stream()
										.filter(plt -> plt.m > label)
										.count();
								return (entry.getValue()
										.stream()
										.reduce(0.0,
												(sum, plt) -> sum += averageFmeasureCache.get(plt),
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
		ExecutorService executor = Executors.newWorkStealingPool();
		List<Callable<int[]>> tasks = new ArrayList<Callable<int[]>>();

		for (PLT plt : plts) {
			tasks.add(() -> {
				logger.info("Getting predictions from " + plt);
				return plt.getTopkLabels(x, k);
			});
		}

		List<int[]> predictions = null;

		try {
			predictions = executor.invokeAll(tasks)
					.stream()
					.map(future -> {
						try {
							return future.get();
						} catch (InterruptedException | ExecutionException e) {
							throw new IllegalStateException(e);
						}
					})
					.collect(Collectors.toList());

			executor.shutdown();
			executor.awaitTermination(Integer.MAX_VALUE, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			logger.warn(
					"Threds interrupted. Prediction took more than " + Integer.MAX_VALUE + " " + TimeUnit.MILLISECONDS
							+ " to finish.");
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

	private void onPLTDiscarded(PLT plt) {
		PLTDiscardedEventArgs args = new PLTDiscardedEventArgs();
		args.discardedPLT = plt;

		pltDiscardedListeners.stream()
				.forEach(listener -> listener.onPLTDiscarded(this, args));
	}
}