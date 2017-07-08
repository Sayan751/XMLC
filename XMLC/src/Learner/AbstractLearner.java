package Learner;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Stopwatch;
import com.google.common.math.Stats;
import com.google.common.primitives.Ints;

import Data.AVPair;
//import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.Instance;
import IO.DataManager;
import event.args.InstanceProcessedEventArgs;
import event.args.InstanceTestedEventArgs;
import event.listeners.IInstanceProcessedListener;
import event.listeners.IInstanceTestedListener;
import interfaces.IFmeasureObserver;
import threshold.ThresholdTuner;
import util.Constants.LearnerDefaultValues;
import util.Constants.LearnerInitProperties;
import util.Constants.ThresholdTuningDataKeys;
import util.IoUtils;
import util.LearnerInitConfiguration;

public abstract class AbstractLearner implements Serializable {
	private static final long serialVersionUID = -1399552145906714507L;

	private static Logger logger = LoggerFactory.getLogger(AbstractLearner.class);

	/**
	 * Number of labels.
	 */
	protected int m = 0;
	/**
	 * Number of features.
	 */
	protected int d = 0;

	/**
	 * Number of training instances seen.
	 */
	protected int nTrain = 0;
	/**
	 * Number of test instances seen.
	 */
	protected int nTest = 0;

	transient protected Properties properties = null;
	protected double[] thresholds = null;

	protected ThresholdTuner thresholdTuner;
	protected ThresholdTuner testTuner;
	protected ThresholdTuner testTopKTuner;

	/**
	 * Holds fmeasures per training instance.
	 * 
	 */
	protected List<Double> fmeasures;
	/**
	 * Holds prequential (before training) fmeasures per training instance.
	 * 
	 */
	protected List<Double> prequentialFmeasures;

	protected boolean isToComputeFmeasureOnTopK;

	/**
	 * Default {@code k} is 'topK' based methods.
	 */
	protected int defaultK;

	transient protected Set<IInstanceTestedListener> instanceTestedListeners;
	transient protected Set<IInstanceProcessedListener> instanceProcessedListeners;
	transient public IFmeasureObserver fmeasureObserver;

	public boolean fmeasureObserverAvailable;

	private UUID id;
	protected boolean shuffleLabels;
	protected boolean measureTime = false;

	transient private Stopwatch stopwatch;
	/**
	 * Total time (in micro seconds) spent in training.
	 */
	protected long totalTrainTime = 0;
	/**
	 * Total time (in micro seconds) spent in testing.
	 */
	protected long totalTestTime = 0;
	/**
	 * Total time (in micro seconds) spent in prequential evaluation.
	 */
	private long totalPrequentialEvaluationTime = 0;
	/**
	 * Total time (in micro seconds) spent in prequential topk-based evaluation.
	 */
	private long totalPrequentialTopkEvaluationTime = 0;
	/**
	 * Total time (in micro seconds) spent in post-training evaluation
	 * (evaluation on training data).
	 */
	private long totalEvaluationTime = 0;
	/**
	 * Total time (in micro seconds) spent in post-training topk-based
	 * evaluation (evaluation on training data).
	 */
	private long totalTopkEvaluationTime = 0;

	// abstract functions
	public abstract void allocateClassifiers(DataManager data);

	public abstract void train(DataManager data);

	// public abstract Evaluator test( AVTable data );
	public abstract double getPosteriors(AVPair[] x, int label);

	public void savemodel(String fname) throws IOException {
		IoUtils.serialize(this, Paths.get(fname));
	}

	public static AbstractLearner loadmodel(String fname)
			throws FileNotFoundException, ClassNotFoundException, IOException {
		return (AbstractLearner) IoUtils.deserialize(Paths.get(fname));
	}

	public int getPrediction(AVPair[] x, int label) {
		if (this.thresholds[label] <= getPosteriors(x, label)) {
			return 1;
		} else {
			return 0;
		}
	}

	public void printParameters() {
		logger.info("Number of labels: " + this.m);
		logger.info("Number of features: " + this.d);
	}

	public static AbstractLearner learnerFactory(Properties properties) {
		AbstractLearner learner = null;

		String learnerName = properties.getProperty("Learner");
		logger.info("--> Learner: {}", learnerName);

		if (learnerName.compareTo("Constant") == 0)
			learner = new ConstantLearner(properties);
		else if (learnerName.compareTo("PLT") == 0)
			learner = new PLT(properties);
		else if (learnerName.compareTo("MLL") == 0)
			learner = new MLL(properties);
		else if (learnerName.compareTo("DeepPLT") == 0)
			learner = new DeepPLT(properties);
		else {
			System.err.println("Unknown learner");
			System.exit(-1);
		}

		return learner;

	}

	// public void tuneThreshold( ThresholdTuning t, DataManager data ){
	// this.setThresholds(t.validate(data, this));
	// }

	public void setThresholds(double[] t) {
		for (int j = 0; j < t.length; j++) {
			this.thresholds[j] = t[j];
		}
	}

	public void setThresholds(double t) {
		for (int j = 0; j < this.thresholds.length; j++) {
			this.thresholds[j] = t;
		}
	}

	public void setThreshold(int label, double t) {
		this.thresholds[label] = t;
	}

	public AbstractLearner() {
		instanceProcessedListeners = new HashSet<IInstanceProcessedListener>();
		instanceTestedListeners = new HashSet<IInstanceTestedListener>();
	}

	public AbstractLearner(Properties properties) {
		this();
		this.properties = properties;

		Object tempPropValue = properties.get(LearnerInitProperties.isToComputeFmeasureOnTopK);
		isToComputeFmeasureOnTopK = tempPropValue != null
				? (Boolean) tempPropValue
				: LearnerDefaultValues.isToComputeFmeasureOnTopK;

		tempPropValue = properties.get(LearnerInitProperties.defaultK);
		defaultK = tempPropValue != null
				? (Integer) tempPropValue
				: LearnerDefaultValues.defaultK;

		fmeasureObserver = (IFmeasureObserver) properties.get(LearnerInitProperties.fmeasureObserver);

		fmeasureObserverAvailable = fmeasureObserver != null;

		if (fmeasureObserverAvailable)
			addInstanceProcessedListener(fmeasureObserver);

		shuffleLabels = Boolean.parseBoolean(
				properties.getProperty(LearnerInitProperties.shuffleLabels,
						String.valueOf(LearnerDefaultValues.shuffleLabels)));
	}

	public AbstractLearner(LearnerInitConfiguration configuration) {
		this();
		isToComputeFmeasureOnTopK = configuration.isToComputeFmeasureOnTopK();
		defaultK = configuration.getDefaultK();
		shuffleLabels = configuration.isToShuffleLabels();
		measureTime = configuration.isMeasureTime();

		fmeasureObserver = configuration.fmeasureObserver;
		fmeasureObserverAvailable = fmeasureObserver != null;

		if (fmeasureObserverAvailable) {
			addInstanceProcessedListener(fmeasureObserver);
			addInstanceTestedListener(fmeasureObserver);
		}
	}

	// naive implementation checking all labels
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		for (int i = 0; i < this.m; i++) {
			if (this.getPosteriors(x, i) >= this.thresholds[i]) {
				positiveLabels.add(i);
			}
		}

		return positiveLabels;
	}

	// naive implementation checking all labels
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();

		for (int i = 0; i < this.m; i++) {
			double post = getPosteriors(x, i);
			if (this.thresholds[i] <= post) {
				positiveLabels.add(new ComparablePair(post, i));
			}
		}

		return positiveLabels;
	}

	public int[] getTopkLabels(AVPair[] x, int k) {
		PriorityQueue<ComparablePair> pq = new PriorityQueue<ComparablePair>();

		for (int i = 0; i < this.m; i++) {
			double post = this.getPosteriors(x, i);
			pq.add(new ComparablePair(post, i));
		}

		int[] labels = new int[k];
		for (int i = 0; i < k; i++) {
			ComparablePair p = pq.poll();
			labels[i] = p.getValue();
		}

		return labels;
	}

	// public void outputPosteriors( String fname, DataManager data )
	// {
	// try{
	// logger.info( "Saving posteriors (" + fname + ")..." );
	// BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
	// new FileOutputStream(fname)));
	//
	// for(int i = 0; i< data.n; i++ ){
	// for( int j = 0; j < this.m; j++ ){
	// writer.write( j + ":" + this.getPosteriors(data.x[i], j) + " " );
	// }
	// writer.write( "\n" );
	// }
	//
	// writer.close();
	// logger.info( "Done." );
	// } catch (IOException e) {
	// logger.info(e.getMessage());
	// }
	//
	// }

	public HashSet<EstimatePair> getSparseProbabilityEstimates(AVPair[] x, double threshold) {

		HashSet<EstimatePair> positiveLabels = new HashSet<EstimatePair>();

		for (int i = 0; i < this.m; i++) {
			double p = getPosteriors(x, i);
			if (p >= threshold)
				positiveLabels.add(new EstimatePair(i, p));
		}

		return positiveLabels;
	}

	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();

		for (int i = 0; i < this.m; i++) {
			double p = getPosteriors(x, i);
			positiveLabels.add(new EstimatePair(i, p));
		}

		while (positiveLabels.size() >= k) {
			positiveLabels.pollLast();
		}

		return positiveLabels;

	}

	public Properties getProperties() {
		return properties;
	}

	public int getNumberOfLabels() {
		return this.m;
	}

	/**
	 * Modifies thresholds as provided by {@code this.thresholdTuner}
	 * 
	 * Note: This is a potential candidate to move to AbstractLearner; subject
	 * to feasibility check.
	 * 
	 * @param data
	 * 
	 */
	protected void tuneThreshold(DataManager data) {
		thresholdTuner
				.getTunedThresholdsSparse(createTuningData(data))
				.forEach((label, threshold) -> setThreshold(label, threshold));

		// Map<Integer, Double> sparseThresholds = thresholdTuner
		// .getTunedThresholdsSparse(createTuningData(data));
		// for (Entry<Integer, Double> entry : sparseThresholds.entrySet()) {
		// setThreshold(entry.getKey(), entry.getValue());
		// }
	}

	protected void tuneThreshold(Instance instance) {
		thresholdTuner
				.getTunedThresholdsSparse(createTuningData(instance))
				.forEach((label, threshold) -> setThreshold(label, threshold));
	}

	/**
	 * 
	 * @param data
	 * @return
	 */
	protected Map<String, Object> createTuningData(DataManager data) {
		Map<String, Object> retVal = new HashMap<String, Object>();
		data.reset();
		switch (thresholdTuner.getTunerType()) {
		case OfoFast:
		case AdaptiveOfoFast:
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();			
			
			while (data.hasNext()) {
				Instance instance = data.getNextInstance();
				trueLabels.add(new HashSet<Integer>(Ints.asList(instance.y)));
				predictedLabels.add(getPositiveLabels(instance.x));
			}
			
			logger.info("Before tuning threshold. Ground truth" + trueLabels + ", predictions: " + predictedLabels);
			
			retVal.put(ThresholdTuningDataKeys.trueLabels, trueLabels);
			retVal.put(ThresholdTuningDataKeys.predictedLabels, predictedLabels);
			break;
		default:
			logger.warn("createTuningData for " + thresholdTuner.getTunerType() + " is not yet implemented.");
			break;

		}
		data.reset();
		return retVal;
	}

	/**
	 * 
	 * @param instance
	 * @return
	 */
	protected Map<String, Object> createTuningData(Instance instance) {
		Map<String, Object> retVal = new HashMap<String, Object>();
		switch (thresholdTuner.getTunerType()) {
		case OfoFast:
		case AdaptiveOfoFast:
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

			trueLabels.add(new HashSet<Integer>(Ints.asList(instance.y)));
			predictedLabels.add(getPositiveLabels(instance.x));

			retVal.put(ThresholdTuningDataKeys.trueLabels, trueLabels);
			retVal.put(ThresholdTuningDataKeys.predictedLabels, predictedLabels);
			break;
		default:
			logger.warn("createTuningData for " + thresholdTuner.getTunerType() + " is not yet implemented.");
			break;

		}
		return retVal;
	}

	public double getAverageFmeasure(boolean isPrequential, boolean isTopk) {
		return fmeasureObserverAvailable ? fmeasureObserver.getAverageFmeasure(this, isPrequential, isTopk)
				: Stats.meanOf(isPrequential ? prequentialFmeasures : fmeasures);
	}

	protected double getFmeasureForInstance(Instance instance, boolean isToPublishFmeasure, boolean isPrequential,
			boolean returnTopKFm) {

		List<Integer> truePositives = Ints.asList(instance.y);

		if (isToPublishFmeasure) {

			if (measureTime) {
				getStopwatch().reset();
				getStopwatch().start();
			}
			HashSet<Integer> predictedPositives = getPositiveLabels(instance.x);
			if (measureTime) {
				getStopwatch().stop();
				if (isPrequential)
					totalPrequentialEvaluationTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
				else
					totalEvaluationTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
			}

			if (measureTime) {
				getStopwatch().reset();
				getStopwatch().start();
			}
			int[] predictedTopkPositives = getTopkLabels(instance.x, defaultK);
			if (measureTime) {
				getStopwatch().stop();
				if (isPrequential)
					totalPrequentialTopkEvaluationTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
				else
					totalTopkEvaluationTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
			}

			double fmeasure = computeFmeasure(truePositives, predictedPositives);
			double topkFmeasure = computeFmeasure(truePositives, Ints.asList(predictedTopkPositives));

			onInstanceProcessed(new InstanceProcessedEventArgs(instance, fmeasure, topkFmeasure, isPrequential));

			return returnTopKFm ? topkFmeasure : fmeasure;

		} else {
			return computeFmeasure(truePositives, returnTopKFm
					? new HashSet<Integer>(Ints.asList(getTopkLabels(instance.x, defaultK)))
					: getPositiveLabels(instance.x));
		}
	}

	protected double getFmeasureForInstance(Instance instance, boolean isToPublishFmeasure, boolean isPrequential) {
		return getFmeasureForInstance(instance, isToPublishFmeasure, isPrequential, isToComputeFmeasureOnTopK);
	}

	protected double getFmeasureForInstance(Instance instance) {
		return getFmeasureForInstance(instance, false, false);
	}

	protected double computeFmeasure(Collection<Integer> truePositives, Collection<Integer> predictedPositives) {
		Set<Integer> intersection = new HashSet<Integer>(truePositives);
		intersection.retainAll(predictedPositives);

		return (2.0 * intersection.size()) / (double) (truePositives.size() + predictedPositives.size());
	}

	/**
	 * @return the numberOfTrainingInstancesSeen
	 */
	public int getnTrain() {
		return nTrain;
	}

	public int getnTest() {
		return nTest;
	}

	public void addInstanceProcessedListener(IInstanceProcessedListener listener) {
		instanceProcessedListeners.add(listener);
	}

	public void removeInstanceProcessedListener(IInstanceProcessedListener listener) {
		instanceProcessedListeners.remove(listener);
	}

	private void onInstanceProcessed(InstanceProcessedEventArgs args) {
		instanceProcessedListeners.forEach(listener -> listener.onInstanceProcessed(this, args));
	}

	protected void evaluate(DataManager data, boolean isPrequential) {
		data.reset();
		while (data.hasNext() == true) {
			evaluate(data.getNextInstance(), isPrequential);
		}
		data.reset();
	}

	protected void evaluate(Instance instance, boolean isPrequential) {
		if (fmeasureObserverAvailable) {
			getFmeasureForInstance(instance, true, isPrequential);
		} else {
			if (isPrequential)
				prequentialFmeasures.add(getFmeasureForInstance(instance));
			else
				fmeasures.add(getFmeasureForInstance(instance));
		}
	}

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public void test(DataManager testData) {
		while (testData.hasNext()) {
			test(testData.getNextInstance());
		}
	}

	public void test(Instance instance) {

		if (measureTime) {
			getStopwatch().reset();
			getStopwatch().start();
		}

		int[] topkPredictedPositives = getTopkLabels(instance.x, defaultK);
		HashSet<Integer> predictedPositives = getPositiveLabels(instance.x);

		if (measureTime) {
			getStopwatch().stop();
			totalTestTime += getStopwatch().elapsed(TimeUnit.MICROSECONDS);
		}

		List<Integer> truePositives = Ints.asList(instance.y);

		predictedOnTestInstance(instance, predictedPositives, topkPredictedPositives);

		double fmeasure = computeFmeasure(truePositives, predictedPositives);
		double topkFmeasure = computeFmeasure(truePositives, Ints.asList(topkPredictedPositives));

		logger.info("Prediction info on test instance - true positives: " + truePositives + ", predicted positves: "
				+ predictedPositives + " (fm: " + fmeasure + "), predicted topk: "
				+ Arrays.toString(topkPredictedPositives) + " (fm: " + topkFmeasure + ")");

		InstanceTestedEventArgs args = new InstanceTestedEventArgs();
		args.instance = instance;
		args.fmeasure = fmeasure;
		args.topkFmeasure = topkFmeasure;

		onInstanceTested(args);

		nTest++;
	}

	/**
	 * Workflow step function; invoked when general and top k predictions are
	 * computed for {@code instance}.
	 * 
	 * @param instance
	 * @param predictedPositives
	 * @param topkPredictedPositives
	 */
	protected void predictedOnTestInstance(Instance instance, HashSet<Integer> predictedPositives,
			int[] topkPredictedPositives) {

		Map<String, Object> tuningData = new HashMap<String, Object>();
		List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
		List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

		trueLabels.add(new HashSet<Integer>(Ints.asList(instance.y)));
		predictedLabels.add(predictedPositives);

		tuningData.put(ThresholdTuningDataKeys.trueLabels, trueLabels);
		tuningData.put(ThresholdTuningDataKeys.predictedLabels, predictedLabels);

		if (testTuner != null)
			testTuner.getTunedThresholdsSparse(tuningData);

		predictedLabels.clear();

		predictedLabels.add(new HashSet<Integer>(Ints.asList(topkPredictedPositives)));

		if (testTopKTuner != null)
			testTopKTuner.getTunedThresholdsSparse(tuningData);
	}

	private void onInstanceTested(InstanceTestedEventArgs args) {
		instanceTestedListeners.forEach(listener -> listener.onInstanceTested(this, args));
	}

	public void addInstanceTestedListener(IInstanceTestedListener listener) {
		instanceTestedListeners.add(listener);
	}

	public void removeInstanceProcessedListener(IInstanceTestedListener listener) {
		instanceTestedListeners.remove(listener);
	}

	public double getMacroFmeasure() {
		return thresholdTuner.getMacroFmeasure();
	}

	public double getTestMacroFmeasure(boolean isTopk) {
		return isTopk ? testTopKTuner.getMacroFmeasure() : testTuner.getMacroFmeasure();
	}

	public double getTestAverageFmeasure(boolean isTopk) {
		return fmeasureObserver.getTestAverageFmeasure(this, isTopk);
	}

	/**
	 * @return the stopwatch
	 */
	public Stopwatch getStopwatch() {
		if (stopwatch == null)
			stopwatch = Stopwatch.createUnstarted();
		return stopwatch;
	}

	/**
	 * @return Average time spent in training (in micro seconds).
	 */
	public double getAverageTrainTime() {
		return nTrain > 0 ? (double) totalTrainTime / (double) nTrain : 0;
	}

	/**
	 * @return Average time spent in testing (in micro seconds).
	 */
	public double getAverageTestTime() {
		return nTest > 0 ? (double) totalTestTime / (double) nTest : 0;
	}

	public double getAverageEvaluationTime(boolean isPrequential, boolean isTopK) {
		if (nTrain <= 0)
			return 0;
		if (isPrequential) {
			return isTopK ? ((double) totalPrequentialTopkEvaluationTime / (double) nTrain)
					: ((double) totalPrequentialEvaluationTime / (double) nTrain);
		}
		return isTopK ? ((double) totalTopkEvaluationTime / (double) nTrain)
				: ((double) totalEvaluationTime / (double) nTrain);
	}
}
