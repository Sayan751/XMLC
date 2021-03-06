package Learner;


import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.TreeSet;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Ints;

import Data.AVPair;
//import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.Instance;
import IO.DataManager;
import threshold.ThresholdTuner;
import util.IoUtils;
import util.Constants.ThresholdTuningDataKeys;


public abstract class AbstractLearner implements Serializable{
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


	transient protected Properties properties = null;
	protected double[] thresholds = null;
	
	protected ThresholdTuner thresholdTuner;

	// abstract functions
	public abstract void allocateClassifiers( DataManager data );
	public abstract void train( DataManager data );
	//public abstract Evaluator test( AVTable data );
	public abstract double getPosteriors(AVPair[] x, int label);

	public void savemodel(String fname ) throws IOException{
		IoUtils.serialize(this, Paths.get(fname));
	}
	public static AbstractLearner loadmodel(String fname ) throws FileNotFoundException, ClassNotFoundException, IOException{
		return (AbstractLearner) IoUtils.deserialize(Paths.get(fname));
	}

	public int getPrediction(AVPair[] x, int label){
		if ( this.thresholds[label] <= getPosteriors(x, label) ) {
			return 1;
		} else {
			return 0;
		}
	}

	public void printParameters() {
		logger.info("Number of labels: " + this.m);
		logger.info("Number of features: " + this.d);
	}
	
	public static AbstractLearner learnerFactory( Properties properties ) {
		AbstractLearner learner = null;
		
		String learnerName = properties.getProperty("Learner");
		logger.info("--> Learner: {}", learnerName);
		
		if (learnerName.compareTo( "Constant" ) == 0)
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
		
//	public void tuneThreshold( ThresholdTuning t, DataManager data ){
//		this.setThresholds(t.validate(data, this));
//	}

	public void setThresholds(double[] t) {		
		for(int j = 0; j < t.length; j++) {
			this.thresholds[j] = t[j];
		}		
	}

	public void setThresholds(double t) {		
		for(int j = 0; j < this.thresholds.length; j++) {
			this.thresholds[j] = t;
		}		
	}
	
	public void setThreshold(int label, double t) {
		this.thresholds[label] = t;
	}	
	
	public AbstractLearner(Properties properties){
		this.properties = properties;		
	}


	// naive implementation checking all labels
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		for( int i = 0; i < this.m; i++ ) {
			if (this.getPosteriors(x, i) >= this.thresholds[i]) {
				positiveLabels.add(i);
			}
		}

		return positiveLabels;
	}
	
	// naive implementation checking all labels
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();

		for( int i = 0; i < this.m; i++ ) {
			double post = getPosteriors(x, i);
			if ( this.thresholds[i] <= post ){
				positiveLabels.add(new ComparablePair(post, i ));
			}
		}

		return positiveLabels;
	}
	
	public int[] getTopkLabels(AVPair[] x, int k) {
		PriorityQueue<ComparablePair> pq = new PriorityQueue<ComparablePair>();
		
		for( int i = 0; i < this.m; i++ ) {
			double post = this.getPosteriors(x, i);
			pq.add(new ComparablePair(post, i));
		}
		
		
		int[] labels = new int[k];
		for( int i=0; i<k; i++ ){
			ComparablePair p = pq.poll();
			labels[i] = p.getValue();
		}
		
		return labels;
	}
	
	
//	public void outputPosteriors( String fname, DataManager data )
//	{
//		try{
//			logger.info( "Saving posteriors (" + fname + ")..." );
//			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
//			          new FileOutputStream(fname)));
//			    
//			for(int i = 0; i< data.n; i++ ){
//				for( int j = 0; j < this.m; j++ ){
//					writer.write( j + ":" + this.getPosteriors(data.x[i], j) + " " );
//				}
//				writer.write( "\n" );
//			}
//						
//			writer.close();
//			logger.info( "Done." );
//		} catch (IOException e) {
//			logger.info(e.getMessage());
//		}
//		
//	}
	
	public HashSet<EstimatePair> getSparseProbabilityEstimates(AVPair[] x, double threshold) {
		
		HashSet<EstimatePair> positiveLabels = new HashSet<EstimatePair>();
		
		for(int i = 0; i < this.m; i++) {
			double p = getPosteriors(x, i);
			if (p >= threshold) 
				positiveLabels.add(new EstimatePair(i, p));
		}
		
		return positiveLabels;
	}
	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();
		
		for(int i = 0; i < this.m; i++) {
			double p = getPosteriors(x, i);			 
			positiveLabels.add(new EstimatePair(i, p));
		}
		
		while( positiveLabels.size()>=k){			
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
		try {
			Map<Integer, Double> sparseThresholds = thresholdTuner
					.getTunedThresholdsSparse(createTuningData(data));
			for (Entry<Integer, Double> entry : sparseThresholds.entrySet()) {
				setThreshold(entry.getKey(), entry.getValue());
			}
		} catch (Exception e) {
			logger.error("Error during tuning the threshlds.", e);
			e.printStackTrace();
		}
	}

	/**
	 * Note: This is a potential candidate to move to AbstractLearner; subject to feasibility
	 * check.
	 * @param data
	 * @return
	 */
	protected Map<String, Object> createTuningData(DataManager data) {
		Map<String, Object> retVal = new HashMap<String, Object>();
		switch (thresholdTuner.getTunerType()) {
		case OfoFast:
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();
			while (data.hasNext()) {
				Instance instance = data.getNextInstance();
				trueLabels.add(new HashSet<Integer>(Ints.asList(instance.y)));
				predictedLabels.add(getPositiveLabels(instance.x));
			}
			retVal.put(ThresholdTuningDataKeys.trueLabels, trueLabels);
			retVal.put(ThresholdTuningDataKeys.predictedLabels, predictedLabels);
			break;
		default:
			logger.warn("createTuningData for " + thresholdTuner.getTunerType() + " is not yet implemented.");
			break;

		}
		return retVal;
	}
	
}
