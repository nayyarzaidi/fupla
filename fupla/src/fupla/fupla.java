package fupla;

import java.util.Arrays;

import weka.classifiers.AbstractClassifier;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

import weka.filters.supervised.attribute.Discretize;

public class fupla extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;

	public xxyDist xxyDist_;

	private Discretize m_Disc = null;

	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private int m_KDB = 1; 											// -K
	
	private boolean m_MVerb = false; 							// -V
	private boolean m_Discretization = false; 				// -D 

	private double[] probs;
	
	int m_BestK_ = 0; 
	int m_BestattIt = 0;

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new Discretize();
			m_Disc.setInputFormat(instances);
			instances = weka.filters.Filter.useFilter(instances, m_Disc);
		}

		// remove instances with missing class
		m_Instances = new Instances(instances);
		m_Instances.deleteWithMissingClass();
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		probs = new double[nc];
		nInstances = m_Instances.numInstances();

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i;
		}
		
		m_BestK_ = m_KDB; 
		m_BestattIt = nAttributes;

		/* 
		 * ------------------------------------------------------
		 * Pass No. 1 (NB)
		 * ------------------------------------------------------
		 */

		xxyDist_ = new xxyDist(m_Instances);
		xxyDist_.addToCount(m_Instances);

		double[] mi = null;
		double[][] cmi = null;

		mi = new double[nAttributes];
		cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);
		
		//System.out.println(Arrays.toString(mi));
		//for (int i = 0; i < nAttributes; i++) {
		//	System.out.println(Arrays.toString(cmi[i]));
		//}
		
		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nK = Math.min(u, m_KDB);
			if (nK > 0) {
				m_Parents[u] = new int[nK];

				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}
				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nK; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// Print the structure
		System.out.println(Arrays.toString(m_Order));
		for (int i = 0; i < nAttributes; i++) {
			System.out.print(i + " : ");
			if (m_Parents[i] != null) {
				for (int j = 0; j < m_Parents[i].length; j++) {
					System.out.print(m_Parents[i][j] + ",");
				}
			}
			System.out.println();
		}
		
		System.out.println("**********************************************");
		System.out.println("First Pass Finished");
		System.out.println("**********************************************");

		/* 
		 * ------------------------------------------------------
		 * Pass No. 2 (KDB)
		 * ------------------------------------------------------
		 */

		dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order, m_Parents, 1);

		// Update dTree_ based on parents
		for (int i = 0; i < nInstances; i++) {
			Instance instance = getM_Instances().instance(i);
			dParameters_.update(instance);
		}
		
		dParameters_.countsToProbability();
		
		System.out.println("**********************************************");
		System.out.println("Second Pass Finished");
		System.out.println("**********************************************");

		/* 
		 * ------------------------------------------------------
		 * Pass No. 3 (SKDB)
		 * ------------------------------------------------------
		 */

		double[][] foldLossFunctallK_ = new double[m_KDB + 1][nAttributes + 1];
		double[][] posteriorDist = new double[m_KDB + 1][nc];

		/* Start the third costly pass through the data */
		for (int i = 0; i < nInstances; i++) {
			Instance instance = getM_Instances().instance(i);
			int x_C = (int)instance.classValue();

			for (int y = 0; y < nc; y++) {
				posteriorDist[0][y] = dParameters_.ploocv(y, x_C); 
			}
			SUtils.normalize(posteriorDist[0]);

			double error = 1.0 - posteriorDist[0][x_C];
			foldLossFunctallK_[0][nAttributes] += error * error;

			for (int k = 1; k <= m_KDB; k++) {
				for (int y = 0; y < nc; y++){ 
					posteriorDist[k][y] = posteriorDist[0][y];
				}
				foldLossFunctallK_[k][nAttributes] += error * error;
			}

			for (int u = 0; u < nAttributes; u++) {

				dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], instance, m_KDB); //Discounting inst from counts

				for (int k = 0; k <= m_KDB; k++)
					SUtils.normalize(posteriorDist[k]);

				for (int k = 0; k <= m_KDB; k++){
					error = 1.0 - posteriorDist[k][x_C];
					foldLossFunctallK_[k][u] += error * error;
				}

			}	
		}
		
		/* Start the book keeping, select the best k and best attributes */
		for (int k = 0; k <= m_KDB; k++) {
			System.out.println("k = " + k);
			for (int u = 0; u < nAttributes; u++){
				System.out.print(foldLossFunctallK_[k][u] + ", ");
			}
			System.out.println(foldLossFunctallK_[k][nAttributes]);
		}

		//Proper kdb selective (RMSE)      
		for (int k = 0; k <= m_KDB; k++) {
			for (int att = 0; att < nAttributes+1; att++) {
				foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att]/nInstances);
			}
			foldLossFunctallK_[k][nAttributes] = foldLossFunctallK_[0][nAttributes]; //The prior is the same for all values of k_
		}

		double globalmin = foldLossFunctallK_[0][nAttributes];

		for (int u = 0; u < nAttributes; u++){
			for (int k = 0; k <= m_KDB; k++) {
				if (foldLossFunctallK_[k][u] < globalmin) {
					globalmin = foldLossFunctallK_[k][u];
					m_BestattIt = u;
					m_BestK_ = k;
				}
			}
		}
		
		m_BestattIt += 1;

		for (int k = 0; k <= m_KDB; k++) {
			System.out.println("k = " + k);
			for (int u = 0; u < nAttributes; u++){
				System.out.print(foldLossFunctallK_[k][u] + ", ");
			}
			System.out.println(foldLossFunctallK_[k][nAttributes]);
		}
		
		System.out.println("Number of features selected is: " + m_BestattIt);
		System.out.println("best k is: " + m_BestK_);
		
		System.out.println("**********************************************");
		System.out.println("Third Pass Finished");
		System.out.println("**********************************************");

		/* 
		 * ------------------------------------------------------
		 * Pass No. 4 (SGD pass)
		 * ------------------------------------------------------
		 */
		
	}

	@Override
	public double[] distributionForInstance(Instance instance) {

		double[] probs = null;

		probs = logDistributionForInstance_MAP(instance);

		SUtils.exp(probs);
		return probs;
	}

	public double[] logDistributionForInstance_MAP(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassProbabilities()[c]; //xxyDist_.xyDist_.pp(c);
		}

		for (int u = 0; u < m_BestattIt; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u, m_BestK_);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYProbability((int) instance.value(m_Order[u]),	c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;	
	}
	
//	public double[] logDistributionForInstance_MAP(Instance instance) {
//
//		double[] probs = new double[nc];
//
//		for (int c = 0; c < nc; c++) {
//			probs[c] = Math.log(SUtils.MEsti(dParameters_.getClassCounts()[c], nInstances, nc));
//		}
//
//		for (int u = 0; u < m_BestattIt; u++) {
//			wdBayesNode bNode = dParameters_.getBayesNode(instance, u, m_BestK_);
//			for (int c = 0; c < nc; c++) {
//				probs[c] += bNode.updateClassDistribution((int) instance.value(m_Order[u]),	c);
//			}
//		}
//
//		SUtils.normalizeInLogDomain(probs);
//		return probs;	
//	}


	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);
		m_Discretization = Utils.getFlag('D', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new fupla(), argv);
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public wdBayesParametersTree getdParameters_() {
		return dParameters_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}

}
