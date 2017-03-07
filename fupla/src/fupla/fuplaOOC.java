package fupla;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;
import optimize.Minimizer;
import optimize.MinimizerTron;
import optimize.DifferentiableFunction;
import optimize.FunctionValues;

import optimize.Result;
import optimize.StopConditions;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

public class fuplaOOC {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances = null;
	private Instances structure;

	int N;
	int n;
	int nc;
	public int[] paramsPerAtt;

	public xxyDist xxyDist_;
	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private boolean m_MVerb = false; 							// -V
	private boolean m_DoSKDB = false;						// -S
	private boolean m_DoDiscriminative = false;			// -D
	private int m_KDB = 1; 											// -K
	private String m_O = "adagrad";								// -S

	private double[] probs;

	int m_BestK_ = 0; 
	int m_BestattIt = 0;
	
	private RandomGenerator rg = null;
	private static final int BUFFER_SIZE = 100000;

	ObjectiveFunction function_to_optimize;

	public void buildClassifier(File sourceFile) throws Exception {
		
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		// remove instances with missing class
		n = structure.numAttributes() - 1;
		nc = structure.numClasses();
		N = structure.numInstances();

		probs = new double[nc];

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = structure.attribute(u).numValues();
		}

		m_Parents = new int[n][];
		m_Order = new int[n];
		for (int i = 0; i < n; i++) {
			m_Order[i] = i;
		}

		m_BestK_ = m_KDB; 
		m_BestattIt = n;

		/*
		 *  ------------------------------------------------------
		 * Pass No. 1
		 * ------------------------------------------------------ 
		 */

		xxyDist_ = new xxyDist(structure);
		Instance row;
		N = 0;
		
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		while ((row = reader.readInstance(structure)) != null) {
			xxyDist_.update(row);
			N++;
		}		
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
		double[] mi = new double[n];
		double[][] cmi = new double[n][n];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < n; u++) {
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
		for (int i = 0; i < n; i++) {
			System.out.print(i + " : ");
			if (m_Parents[i] != null) {
				for (int j = 0; j < m_Parents[i].length; j++) {
					System.out.print(m_Parents[i][j] + ",");
				}
			}
			System.out.println();
		}

		/*
		 *  ------------------------------------------------------
		 * Pass No. 2
		 * ------------------------------------------------------ 
		 */

		dParameters_ = new wdBayesParametersTree(n, nc, paramsPerAtt, m_Order, m_Parents, 1);
		
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		
		while ((row = reader.readInstance(structure)) != null) {
			dParameters_.update(row);
		}
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
		dParameters_.countsToProbability();

		//System.out.println(dParameters_.getNLL_MAP(m_Instances));

		if (m_DoSKDB) {

			/*
			 * ------------------------------------------------------
			 * Pass No. 3 (SKDB)
			 * ------------------------------------------------------
			 */

			double[][] foldLossFunctallK_ = new double[m_KDB + 1][n + 1];
			double[][] posteriorDist = new double[m_KDB + 1][nc];

			/* Start the third costly pass through the data */
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			for (int i = 0; i < N; i++) {
				Instance instance = m_Instances.instance(i);
				int x_C = (int)instance.classValue();

				for (int y = 0; y < nc; y++) {
					posteriorDist[0][y] = dParameters_.ploocv(y, x_C); 
				}
				SUtils.normalize(posteriorDist[0]);

				double error = 1.0 - posteriorDist[0][x_C];
				foldLossFunctallK_[0][n] += error * error;

				for (int k = 1; k <= m_KDB; k++) {
					for (int y = 0; y < nc; y++){ 
						posteriorDist[k][y] = posteriorDist[0][y];
					}
					foldLossFunctallK_[k][n] += error * error;
				}

				for (int u = 0; u < n; u++) {

					dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], instance, m_KDB); //Discounting inst from counts

					for (int k = 0; k <= m_KDB; k++)
						SUtils.normalize(posteriorDist[k]);

					for (int k = 0; k <= m_KDB; k++){
						error = 1.0 - posteriorDist[k][x_C];
						foldLossFunctallK_[k][u] += error * error;
					}

				}	
			}
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			/* Start the book keeping, select the best k and best attributes */
			for (int k = 0; k <= m_KDB; k++) {
				System.out.println("k = " + k);
				for (int u = 0; u < n; u++){
					System.out.print(foldLossFunctallK_[k][u] + ", ");
				}
				System.out.println(foldLossFunctallK_[k][n]);
			}

			//Proper kdb selective (RMSE)      
			for (int k = 0; k <= m_KDB; k++) {
				for (int att = 0; att < n+1; att++) {
					foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att]/N);
				}
				foldLossFunctallK_[k][n] = foldLossFunctallK_[0][n]; //The prior is the same for all values of k_
			}

			double globalmin = foldLossFunctallK_[0][n];

			for (int u = 0; u < n; u++){
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
				for (int u = 0; u < n; u++){
					System.out.print(foldLossFunctallK_[k][u] + ", ");
				}
				System.out.println(foldLossFunctallK_[k][n]);
			}

			System.out.println("Number of features selected is: " + m_BestattIt);
			System.out.println("best k is: " + m_BestK_);

			/*
			 * Clean-up the data structure free some memory 
			 */
			System.out.println("Cleaning up the data structure");
			dParameters_.cleanUp(m_BestattIt, m_BestK_);
			dParameters_.setNAttributes(m_BestattIt);
			n = m_BestattIt;
		}

		// Allocate dParameters after cleaning-up
		dParameters_.allocate();
		dParameters_.initializeParametersWithVal(1);


		/* 
		 * ------------------------------------------------------
		 * Pass No. 4 (SGD)
		 * ------------------------------------------------------
		 */

		if (m_DoDiscriminative) {

			if (m_O.equalsIgnoreCase("Adagrad")) {

				function_to_optimize = new ObjectiveFunction();

				int maxIterations = 100;
				double eps = 0.0001;
				
				MinimizerTron alg = new MinimizerTron();
				alg.setMaxIterations(maxIterations);
				Result result;	

				System.out.print("fx_Tron_ =  [");
				//result = alg.run(function_to_optimize, dParameters_.getParameters(), eps);
				System.out.println("];");
				//System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();

			} 
		}

	}
	
	class ObjectiveFunction {
		
	}

	public double[] distributionForInstance(Instance inst) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			//probs[c] = dParameters_.getParameters()[c];
			probs[c] = dParameters_.getParameters()[c] * dParameters_.getClassProbabilities()[c];

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				//probs[c] += wd.getXYParameter((int)uval, c);
				probs[c] += wd.getXYParameter((int)uval, c) * wd.getXYProbability((int)uval, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		return probs;
	}

	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		m_DoSKDB = Utils.getFlag('S', options);
		m_DoDiscriminative = Utils.getFlag('D', options);

		Utils.checkForRemainingOptions(options);
	}

	public int getNInstances() {
		return N;
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

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}
	
	public void setM_MVerb(boolean m_MVerb) {
		this.m_MVerb = m_MVerb;
	}

	public int getM_KDB() {
		return m_KDB;
	}

	public void setM_KDB(int m_KDB) {
		this.m_KDB = m_KDB;
	}
	
	public boolean isM_DoSKDB() {
		return m_DoSKDB;
	}

	public void setM_DoSKDB(boolean m_DoSKDB) {
		this.m_DoSKDB = m_DoSKDB;
	}

	public boolean isM_DoDiscriminative() {
		return m_DoDiscriminative;
	}

	public void setM_DoDiscriminative(boolean m_DoDiscriminative) {
		this.m_DoDiscriminative = m_DoDiscriminative;
	}

	public int getnAttributes() {
		return n;
	}
	
	public void setRandomGenerator(RandomGenerator rg) {
		this.rg = rg;
	}

}
