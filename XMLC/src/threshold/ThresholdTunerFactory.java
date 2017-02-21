package threshold;

import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import util.Constants.LearnerInitProperties;

public abstract class ThresholdTunerFactory {
	private static Logger logger = LoggerFactory.getLogger(ThresholdTunerFactory.class);

	public static ThresholdTuner createThresholdTuner(int numberOfLabels, Properties properties) {

		ThresholdTuners type = properties.containsKey(LearnerInitProperties.tunerType)
				? ThresholdTuners.valueOf(properties.getProperty(LearnerInitProperties.tunerType))
				: ThresholdTuners.None;

		ThresholdTunerInitOption initOption = (ThresholdTunerInitOption) properties
				.get(LearnerInitProperties.tunerInitOption);
		return createThresholdTuner(numberOfLabels, type, initOption);
	}

	public static ThresholdTuner createThresholdTuner(int numberOfLabels, ThresholdTuners type,
			ThresholdTunerInitOption initOption) {
		ThresholdTuner retVal = null;

		switch (type) {
		case OfoFast:
			retVal = new OfoFastThresholdTuner(numberOfLabels, initOption);
			break;
		case AdaptiveOfoFast:
			retVal = new AdaptiveOfoFastThresholdTuner(numberOfLabels, initOption);
			break;

		default:
			logger.info("ThresholdTuner implementation for " + type + " is not yet implmented.");
			break;
		}

		return retVal;
	}
}