package fupla;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

import org.apache.commons.math3.random.MersenneTwister;

public class fuplaOOC {

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
	private String m_O = "adagrad";								// -O

	private static boolean m_DoRegularization = false;		  // -R
	private static boolean m_DoAdaptiveRegularization = false;		  // -A

	private static double m_Lambda = 0.001;						  // -L
	private static double m_Eta = 0.01;                                 // -E
	private static boolean m_DoCrossvalidate = false;          // -C

	private static int m_NumIterations = 1;                            // -I
	private static int m_BufferSize = 1;                                  // -B

	private static boolean m_DoWANBIAC = false;               // -W

	double smoothingParameter = 1e-9;

	int m_BestK_ = 0; 
	int m_BestattIt = 0;

	private static final int BUFFER_SIZE = 100000;
	
	private Instances instancesTrain = null;
	private Instances instancesTest = null;

	public void buildClassifier(File sourceFile) throws Exception {

		System.out.println("[----- fuplaOOC -----]: Reading structure -- " + sourceFile);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		// remove instances with missing class
		n = structure.numAttributes() - 1;
		nc = structure.numClasses();
		N = structure.numInstances();

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

		System.out.println("Experiment Configuration");
		System.out.println(" ----------------------------------- ");
		System.out.println("m_KDB = " + m_KDB);
		System.out.println("m_DoSKDB = " + m_DoSKDB);
		System.out.println("m_DoWANBIAC = " + m_DoWANBIAC);
		if (m_DoDiscriminative) {
			System.out.println("m_O = " + m_O);
			System.out.println("Iterations = " + m_NumIterations);
			System.out.println("m_DoRegularization = " + m_DoRegularization);
			if (m_DoRegularization) {
				System.out.println("m_DoAdaptiveRegularization = " + m_DoAdaptiveRegularization);
				System.out.println("m_Lambda = " + m_Lambda);
			}
			System.out.println("m_DoCrossvalidate = " + m_DoCrossvalidate);
		} else {
			System.out.println(" ----------------------------------- ");
		}

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
			xxyDist_.setNoData();
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
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		while ((row = reader.readInstance(structure)) != null) {
			dParameters_.update(row);
		}
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		dParameters_.countsToProbability();

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
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			while ((row = reader.readInstance(structure)) != null)  {
				int x_C = (int)row.classValue();

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

					dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], row, m_KDB); //Discounting inst from counts

					for (int k = 0; k <= m_KDB; k++)
						SUtils.normalize(posteriorDist[k]);

					for (int k = 0; k <= m_KDB; k++){
						error = 1.0 - posteriorDist[k][x_C];
						foldLossFunctallK_[k][u] += error * error;
					}

				}	
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
				System.out.print("k = " + k + ", ");
				for (int u = 0; u < n; u++){
					System.out.print(foldLossFunctallK_[k][u] + ", ");
				}
				System.out.println(foldLossFunctallK_[k][n]);
			}

			if (m_BestattIt > n) 
				m_BestattIt = 0;

			System.out.println("**************************************************************************");
			System.out.println("Number of features selected is: " + m_BestattIt + " out of " + n);
			System.out.println("**************************************************************************");
			System.out.println("Best k is: " + m_BestK_ + " out of " + m_KDB);
			System.out.println("**************************************************************************");

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

		if (m_DoDiscriminative) {
			dParameters_.initializeParametersWithVal(0);
		} else {
			dParameters_.initializeParametersWithVal(1);
		}

		/* 
		 * ------------------------------------------------------
		 * Pass No. 4 (SGD)
		 * ------------------------------------------------------
		 */

		if (m_DoDiscriminative) {

			if (m_O.equalsIgnoreCase("sgd")) {

				doSGD(sourceFile);

			} else if (m_O.equalsIgnoreCase("adaptive")) {

				doAdaptive(sourceFile);

			} else if (m_O.equalsIgnoreCase("adagrad")) {

				//doAdagrad(sourceFile);
				doAdagradOpt(sourceFile);

			} else if (m_O.equalsIgnoreCase("adadelta")) {

				doAdadelta(sourceFile);

			}
		}

		System.out.println("Finished training");
	}

	/*
	 * -------------------------------------------------------------------------------------
	 * Adaptive 
	 * SGD
	 * AdaGrad
	 * AdaDelta
	 * -------------------------------------------------------------------------------------
	 */

	private void doSGD(File sourceFile) throws FileNotFoundException, IOException {

		System.out.println("StepSize = " + m_Eta);
		System.out.println(" ----------------------------------- ");

		ArrayList<Integer> indexList = null;

		if (m_DoCrossvalidate) {

			Instances instancesTrain = null;
			Instances instancesTest = null;

			Instances[] instanceList;
			//instanceList = getTrainTestInstances(sourceFile, N);

			indexList = getTrainTestIndices(N);
			instanceList = getTrainTestInstances(sourceFile, indexList);

			Collections.sort(indexList);

			instancesTrain = instanceList[0];
			instancesTest = instanceList[1];

			System.out.println("Finding Alpha (sgd), Please Wait");
			m_Eta = optimizeAlphaSGD(sourceFile, instancesTrain, instancesTest);
			System.out.println("Using m_Eta (after Cross-validation) = ************ " + m_Eta + " ************");
		}

		System.out.print("fx_SGD = [");

		int np = dParameters_.getNp();

		double f = evaluateFunction(sourceFile);
		System.out.print(f + ", ");

		double[] gradients = new double[np];

		for (int iter = 0; iter < m_NumIterations; iter++) {

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			int t = 0;
			while ((row = reader.readInstance(structure)) != null)  {

				Arrays.fill(gradients, 0);

				//				if (m_DoCrossvalidate) {
				//					if (Collections.binarySearch(indexList, t) >= 0) {
				//						continue;
				//					}
				//				}

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);

				if (m_DoRegularization) {
					//regularizeGradient(gradients);
				}

				double stepSize = m_Eta;
				updateParameters(row, probs, x_C, stepSize, gradients);

				t++;
			}

			f = evaluateFunction(sourceFile);
			System.out.print(f + ", ");
		}
		System.out.println("];");

	}

	private double optimizeAlphaSGD(File sourceFile, Instances instancesTrain, Instances instancesTest) {

		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			System.out.print(".");

			dParameters_.initializeParametersWithVal(0);

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);
				double[] gradients = new double[np];

				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);

				computeGrad(instance, probs, x_C, gradients);

				//dParameters_.updateParameters(alpha[i], gradients);
				updateParameters(instance, probs, x_C, alpha[i], gradients);
			}

			/* Test Classifier  with alpha i */
			double m_RMSE = 0;
			double m_Error = 0;

			for (int ii = 0; ii < instancesTest.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				double[] probs = new double[nc];
				probs = distributionForInstance(instance);
				int x_C = (int) instance.classValue();

				int pred = -1;
				double bestProb = Double.MIN_VALUE;
				for (int y = 0; y < nc; y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) {
							pred = y;
							bestProb = probs[y];
						}
						m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}

				if (pred != x_C) {
					m_Error += 1;
				}
			}

			perf[i] = m_RMSE;
		}

		for (int i = 0; i < alpha.length; i++) {
			System.out.println("Alpha = " + alpha[i] + " -- " + "RMSE = " + perf[i]);
		}

		dParameters_.initializeParametersWithVal(0);

		return alpha[SUtils.minLocationInAnArray(perf)];
	}

	private void doAdaptive(File sourceFile) throws FileNotFoundException, IOException {

		System.out.println("StepSize = " + m_Eta);
		System.out.println(" ----------------------------------- ");

		ArrayList<Integer> indexList = null;

		if (m_DoCrossvalidate) {

			Instances instancesTrain = null;
			Instances instancesTest = null;

			Instances[] instanceList;
			//instanceList = getTrainTestInstances(sourceFile, N);

			indexList = getTrainTestIndices(N);
			instanceList = getTrainTestInstances(sourceFile, indexList);

			Collections.sort(indexList);

			instancesTrain = instanceList[0];
			instancesTest = instanceList[1];

			System.out.println("Finding Alpha (adaptive), Please Wait");
			m_Eta = optimizeAlphaAdaptive(sourceFile, instancesTrain, instancesTest);
			System.out.println("Using m_Eta (after Cross-validation) = " + m_Eta);
		}

		System.out.print("fx_ADAPTIVE = [");

		int np = dParameters_.getNp();

		double f = evaluateFunction(sourceFile);
		System.out.print(f + ", ");

		double[] gradients = new double[np];

		int t = 1;
		for (int iter = 0; iter < m_NumIterations; iter++) {

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				if (m_DoCrossvalidate) {
					if (Collections.binarySearch(indexList, t) >= 0) {
						continue;
					}
				}

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);

				if (m_DoRegularization) {
					//regularizeGradient(gradients);
				}

				if (t % m_BufferSize == 0) {
					double stepSize = (m_DoRegularization) ? (m_Eta / (1 + t)) : (m_Eta / (1 + (m_Lambda * t)));

					//dParameters_.updateParameters(stepSize, gradients);
					updateParameters(row, probs, x_C, stepSize, gradients);

					Arrays.fill(gradients, 0);
				}

				//if (t % 10000 == 0) {
				//	System.out.print(t + ", ");
				//}

				t++;
			}

			f = evaluateFunction(sourceFile);
			System.out.print(f + ", ");
		}
		System.out.println("];");
		System.out.println("Did: " + t + " updates.");

	}

	private double optimizeAlphaAdaptive(File sourceFile, Instances instancesTrain, Instances instancesTest) {
		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			System.out.print(".");

			dParameters_.initializeParametersWithVal(0);

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);
				double[] gradients = new double[np];

				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);

				computeGrad(instance, probs, x_C, gradients);

				double stepSize = (m_DoRegularization) ? (alpha[i] / (1 + ii)) : (alpha[i] / (1 + (m_Lambda * ii)));

				//dParameters_.updateParameters(alpha[i], gradients);
				updateParameters(instance, probs, x_C, stepSize, gradients);
			}

			/* Test Classifier  with alpha i */
			double m_RMSE = 0;
			double m_Error = 0;

			for (int ii = 0; ii < instancesTest.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				double[] probs = new double[nc];
				probs = distributionForInstance(instance);
				int x_C = (int) instance.classValue();

				int pred = -1;
				double bestProb = Double.MIN_VALUE;
				for (int y = 0; y < nc; y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) {
							pred = y;
							bestProb = probs[y];
						}
						m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}

				if (pred != x_C) {
					m_Error += 1;
				}
			}

			perf[i] = m_RMSE;
		}

		for (int i = 0; i < alpha.length; i++) {
			System.out.println("Alpha = " + alpha[i] + " -- " + "RMSE = " + perf[i]);
		}

		dParameters_.initializeParametersWithVal(0);

		return alpha[SUtils.minLocationInAnArray(perf)];
	}

	private void doAdagrad(File sourceFile) throws FileNotFoundException, IOException {

		System.out.println("Eta_0 = " + m_Eta);
		System.out.println("SmoothingParameter = " + smoothingParameter);
		System.out.println(" ----------------------------------- ");

		ArrayList<Integer> indexList = null;

		if (m_DoCrossvalidate) {

			Instances instancesTrain = null;
			Instances instancesTest = null;

			Instances[] instanceList;
			//instanceList = getTrainTestInstances(sourceFile, N);

			indexList = getTrainTestIndices(N);
			instanceList = getTrainTestInstances(sourceFile, indexList);

			Collections.sort(indexList);

			instancesTrain = instanceList[0];
			instancesTest = instanceList[1];

			System.out.println("Finding Alpha (adaptive), Please Wait");
			m_Eta = optimizeAlphaAdagrad(sourceFile, instancesTrain, instancesTest);
			System.out.println("Using m_Eta (after Cross-validation) = " + m_Eta);
		}

		int np = dParameters_.getNp();

		double[] G = new double[np];
		double[] gradients = new double[np];

		System.out.print("fx_ADAGRAD = [");

		double f = evaluateFunction(sourceFile);
		System.out.print(f + ", ");

		int t = 0;
		for (int iter = 0; iter < m_NumIterations; iter++) {

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				Arrays.fill(gradients, 0);

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);

				if (m_DoRegularization) {
					//regularizeGradient(gradients);
				}

				for (int j = 0; j < np; j++) {
					G[j] += ((gradients[j] * gradients[j]));
				}

				double stepSize[] = new double[np];
				for (int j = 0; j < np; j++) {
					stepSize[j] = m_Eta / (smoothingParameter + Math.sqrt(G[j]));

					if (stepSize[j] == Double.POSITIVE_INFINITY) {
						stepSize[j] = 0.0;
					}
				}

				//dParameters_.updateParameters(stepSize, gradients);
				updateParameters(row, probs, x_C, stepSize, gradients);

				t++;
			}

			f = evaluateFunction(sourceFile);
			System.out.print(f + ", ");
		}
		System.out.println("];");
		System.out.println("Did: " + t + " updates.");

	}

	private double optimizeAlphaAdagrad(File sourceFile, Instances instancesTrain, Instances instancesTest) {

		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			System.out.print(".");

			dParameters_.initializeParametersWithVal(0);

			double[] G = new double[np];
			double[] gradients = new double[np];

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				Arrays.fill(gradients, 0);

				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);

				computeGrad(instance, probs, x_C, gradients);

				for (int j = 0; j < np; j++) {
					G[j] += ((gradients[j] * gradients[j]));
				}

				double stepSize[] = new double[np];
				for (int j = 0; j < np; j++) {
					stepSize[j] = alpha[i] / (smoothingParameter + Math.sqrt(G[j]));

					if (stepSize[j] == Double.POSITIVE_INFINITY) {
						stepSize[j] = 0.0;
					}
				}

				// Updated parameters
				//dParameters_.updateParameters(stepSize, gradients);
				updateParameters(instance, probs, x_C, stepSize, gradients);
			}

			/* Test Classifier  with alpha i */
			double m_RMSE = 0;
			double m_Error = 0;

			for (int ii = 0; ii < instancesTest.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				double[] probs = new double[nc];
				probs = distributionForInstance(instance);
				int x_C = (int) instance.classValue();

				int pred = -1;
				double bestProb = Double.MIN_VALUE;
				for (int y = 0; y < nc; y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) {
							pred = y;
							bestProb = probs[y];
						}
						m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}

				if (pred != x_C) {
					m_Error += 1;
				}
			}

			perf[i] = m_RMSE;
		}

		for (int i = 0; i < alpha.length; i++) {
			System.out.println("Alpha = " + alpha[i] + " -- " + "RMSE = " + perf[i]);
		}

		dParameters_.initializeParametersWithVal(0);

		return alpha[SUtils.minLocationInAnArray(perf)];
	}

	private void doAdagradOpt(File sourceFile) throws FileNotFoundException, IOException {

		System.out.println("Running an OPTimized version");
		System.out.println("Eta_0 = " + m_Eta);
		System.out.println("SmoothingParameter = " + smoothingParameter);
		System.out.println(" ----------------------------------- ");

		ArrayList<Integer> indexList = null;

		if (m_DoCrossvalidate) {
			
			boolean tempRegularization = m_DoRegularization;
			boolean tempAdaptiveRegularization = m_DoAdaptiveRegularization;
			m_DoRegularization = false;
			m_DoAdaptiveRegularization = false;
			
			Instances[] instanceList;

			indexList = getTrainTestIndices(N);
			instanceList = getTrainTestInstances(sourceFile, indexList);

			Collections.sort(indexList);

			instancesTrain = instanceList[0];
			instancesTest = instanceList[1];

			System.out.println("Finding Alpha (Adagrad), Please Wait");
			//m_Eta = optimizeAlphaAdagrad(sourceFile, instancesTrain, instancesTest);
			//m_Eta = optimizeAlphaAdagradOpt(sourceFile, instancesTrain, instancesTest);
			m_Eta = optimizeAlphaAdagradRecursiveCaller(sourceFile, -6, 6, 10, instancesTrain,  instancesTest);
			System.out.println("Using m_Eta (after Cross-validation) = " + m_Eta);
			
			m_DoRegularization = tempRegularization;
			m_DoAdaptiveRegularization = tempAdaptiveRegularization;
		}

		int np = dParameters_.getNp();

		double[] G = new double[np];

		System.out.print("fx_ADAGRADopt = [");

		double f = evaluateFunction(sourceFile);
		System.out.print(f + ", ");

		int t = 0;
		for (int iter = 0; iter < m_NumIterations; iter++) {

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGradAndUpdateParameters(row, probs, x_C, G);

				t++;
			}

			f = evaluateFunction(sourceFile);
			System.out.print(f + ", ");
		}
		System.out.println("];");
		System.out.println("Did: " + t + " updates.");

	}

	private double optimizeAlphaAdagradOpt(File sourceFile, Instances instancesTrain, Instances instancesTest) {

		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			m_Eta = alpha[i] ;

			System.out.print(".");

			dParameters_.initializeParametersWithVal(0);

			double[] G = new double[np];

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);

				computeGradAndUpdateParameters(instance, probs, x_C, G);
			}

			/* Test Classifier  with alpha i */
			double m_RMSE = 0;
			double m_Error = 0;

			for (int ii = 0; ii < instancesTest.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				double[] probs = new double[nc];
				probs = distributionForInstance(instance);
				int x_C = (int) instance.classValue();

				int pred = -1;
				double bestProb = Double.MIN_VALUE;
				for (int y = 0; y < nc; y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) {
							pred = y;
							bestProb = probs[y];
						}
						m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}

				if (pred != x_C) {
					m_Error += 1;
				}
			}

			perf[i] = m_RMSE;
		}

		for (int i = 0; i < alpha.length; i++) {
			System.out.println("Alpha = " + alpha[i] + " -- " + "RMSE = " + perf[i]);
		}

		dParameters_.initializeParametersWithVal(0);

		return alpha[SUtils.minLocationInAnArray(perf)];
	}

	private double optimizeAlphaAdagradRecursiveCaller(File sourceFile, 
			double low, double high, int scale, Instances instancesTrain, Instances instancesTest) {

		double lowAlphaPerf = 0;
		double highAlphaPerf = 1e20;
		double diff = highAlphaPerf - lowAlphaPerf;

		while (diff > 0.01) {
			System.out.println("Searching between brackets [" + Math.pow(10,low) + ", " + Math.pow(10,high) + "]");
			double[] results = optimizeAlphaAdagradRecursive(sourceFile, low, high, scale,  instancesTrain, instancesTest);
			lowAlphaPerf = results[0];
			highAlphaPerf = results[1];
			low = results[2];
			high = results[3];

			diff = Math.abs(lowAlphaPerf - highAlphaPerf);
		}

		return (Math.pow(10, low) + Math.pow(10, high))/2;
	}

	private double[] optimizeAlphaAdagradRecursive(File sourceFile, 
			double low, double high, int scale, Instances instancesTrain, Instances instancesTest) {

		int np = dParameters_.getNp();

		double binsize = (high - low)/scale;
		double[] alpha = new double[scale + 1];
		alpha[0] = low;
		for (int i = 1; i < scale; i++) {
			alpha[i] = alpha[i-1] + binsize;
		}
		alpha[scale] = high;

		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {
			m_Eta = Math.pow(10, alpha[i]) ;
			System.out.print(".");
			dParameters_.initializeParametersWithVal(0);
			double[] G = new double[np];

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);
				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);
				computeGradAndUpdateParameters(instance, probs, x_C, G);
			}

			/* Test Classifier  with alpha i */
			double m_RMSE = 0;

			for (int ii = 0; ii < instancesTest.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);
				double[] probs = new double[nc];
				probs = distributionForInstance(instance);
				int x_C = (int) instance.classValue();
				double bestProb = Double.MIN_VALUE;
				for (int y = 0; y < nc; y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) { bestProb = probs[y]; }
						m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}
			}
			perf[i] = m_RMSE;
		}

		for (int i = 0; i < alpha.length; i++) {
			System.out.println("Alpha = " + Math.pow(10,alpha[i]) + " -- " + "RMSE = " + perf[i]);
		}

		dParameters_.initializeParametersWithVal(0);

		int center = SUtils.minLocationInAnArray(perf);

		double[] results = new double[4];
		results[0] = perf[center - 1];
		results[1] = perf[center + 1];
		results[2] = alpha[center - 1];
		results[3] = alpha[center + 1];

		return results;
	}

	private void doAdadelta(File sourceFile) throws FileNotFoundException, IOException {

		double rho = m_Eta;

		System.out.println("rho = " + m_Eta);
		System.out.println("SmoothingParameter = " + smoothingParameter);
		System.out.println(" ----------------------------------- ");

		ArrayList<Integer> indexList = null;

		if (m_DoCrossvalidate) {

			Instances instancesTrain = null;
			Instances instancesTest = null;

			Instances[] instanceList;
			//instanceList = getTrainTestInstances(sourceFile, N);

			indexList = getTrainTestIndices(N);
			instanceList = getTrainTestInstances(sourceFile, indexList);

			Collections.sort(indexList);

			instancesTrain = instanceList[0];
			instancesTest = instanceList[1];

			System.out.println("Finding Alpha (adaptive), Please Wait");
			m_Eta = optimizeAlphaAdadelta(sourceFile, instancesTrain, instancesTest);
			System.out.println("Using m_Eta (after Cross-validation) = " + m_Eta);
		}

		int np = dParameters_.getNp();

		double[] G = new double[np];
		double[] D = new double[np];

		double[] gradients = new double[np];

		System.out.print("fx_ADADELTA = [");

		double f = evaluateFunction(sourceFile);
		System.out.print(f + ", ");

		int t = 0;
		for (int iter = 0; iter < m_NumIterations; iter++) {

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
			this.structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			while ((row = reader.readInstance(structure)) != null)  {

				Arrays.fill(gradients, 0);

				int x_C = (int) row.classValue();
				double[] probs = predict(row);
				SUtils.exp(probs);

				computeGrad(row, probs, x_C, gradients);

				if (m_DoRegularization) {
					//regularizeGradient(gradients);
				}

				double stepSize[] = new double[np];

				for (int i = 0; i < np; i++) {
					G[i] = (rho * G[i]) + ((1 - rho) * (gradients[i] * gradients[i]));

					stepSize[i] = - ((Math.sqrt(D[i] + smoothingParameter)) / (Math.sqrt(G[i] + smoothingParameter))) * gradients[i];

					D[i] = (rho * D[i]) + ((1.0 - rho) * (stepSize[i] * stepSize[i]));
				}

				//dParameters_.updateParameters(stepSize, gradients);
				updateParameters(row, probs, x_C, stepSize, gradients);

				t++;
			}

			f = evaluateFunction(sourceFile);
			System.out.print(f + ", ");
		}
		System.out.println("];");
		System.out.println("Did: " + t + " updates.");

	}

	private double optimizeAlphaAdadelta(File sourceFile, Instances instancesTrain, Instances instancesTest) {

		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			double rho = alpha[i];

			System.out.print(".");

			dParameters_.initializeParametersWithVal(0);

			double[] G = new double[np];
			double[] D = new double[np];

			double[] gradients = new double[np];

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				Arrays.fill(gradients, 0);

				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);

				computeGrad(instance, probs, x_C, gradients);

				double stepSize[] = new double[np];
				for (int j = 0; j < np; j++) {
					G[j] = (rho * G[j]) + ((1 - rho) * (gradients[j] * gradients[j]));

					stepSize[j] = - ((Math.sqrt(D[j] + smoothingParameter)) / (Math.sqrt(G[j] + smoothingParameter))) * gradients[j];

					D[j] = (rho * D[j]) + ((1.0 - rho) * (stepSize[j] * stepSize[j]));
				}

				// Updated parameters
				dParameters_.updateParameters(stepSize, gradients);
			}

			/* Test Classifier  with alpha i */
			double m_RMSE = 0;
			double m_Error = 0;

			for (int ii = 0; ii < instancesTest.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);

				double[] probs = new double[nc];
				probs = distributionForInstance(instance);
				int x_C = (int) instance.classValue();

				int pred = -1;
				double bestProb = Double.MIN_VALUE;
				for (int y = 0; y < nc; y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) {
							pred = y;
							bestProb = probs[y];
						}
						m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
					} else {
						System.err.println("probs[ " + y + "] is NaN! oh no!");
					}
				}

				if (pred != x_C) {
					m_Error += 1;
				}
			}

			perf[i] = m_RMSE;
		}

		for (int i = 0; i < alpha.length; i++) {
			System.out.println("Alpha = " + alpha[i] + " -- " + "RMSE = " + perf[i]);
		}

		dParameters_.initializeParametersWithVal(0);

		return alpha[SUtils.minLocationInAnArray(perf)];

	}

	/*
	 * -------------------------------------------------------------------------------------
	 * 5 Important functions
	 * EvaulateFunction
	 * Predict
	 * ComputeGradient
	 * ComputeHessian
	 * distributionForInstance
	 * -------------------------------------------------------------------------------------
	 */

	public double evaluateFunction(File sourceFile) throws IOException {
		double f = 0;
		double mLogNC = - Math.log(nc);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {

			int x_C = (int) row.classValue();
			double[] probs = predict(row);

			f += mLogNC - probs[x_C];
		}

		return f;
	}

	private double[] predict(Instance inst) {
		double[] probs = new double[nc];

		if (m_DoWANBIAC) {
			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c] * dParameters_.getClassProbabilities()[c];


				for (int u = 0; u < n; u++) {
					double uval = inst.value(m_Order[u]);

					wdBayesNode wd = dParameters_.getBayesNode(inst, u);

					probs[c] += wd.getXYParameter((int)uval, c) * wd.getXYProbability((int)uval, c);
				}
			}
		} else {
			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c];


				for (int u = 0; u < n; u++) {
					double uval = inst.value(m_Order[u]);

					wdBayesNode wd = dParameters_.getBayesNode(inst, u);

					probs[c] += wd.getXYParameter((int)uval, c);
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	private void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		if (m_DoWANBIAC) {
			for (int c = 0; c < nc; c++) {
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * dParameters_.getClassProbabilities()[c];
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					gradients[posp] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * wd.getXYProbability((int)uval, c);
				}
			}
		} else {
			for (int c = 0; c < nc; c++) {
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]);
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					gradients[posp] += (-1) * (SUtils.ind(c, x_C) - probs[c]);
				}
			}
		}

	}

	private double computeGradSum(Instance inst, double[] probs, int x_C) {

		double g = 0;
		
		if (m_DoWANBIAC) {
			
			for (int c = 0; c < nc; c++) {
				g += (-1) * (SUtils.ind(c, x_C) - probs[c]) * dParameters_.getClassProbabilities()[c] * 
						(dParameters_.getParameters()[c] - dParameters_.getClassProbabilities()[c]);
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					g += (-1) * (SUtils.ind(c, x_C) - probs[c]) * wd.getXYProbability((int)uval, c) * 
							(wd.getXYParameter((int)uval, c) - wd.getXYProbability((int)uval, c));
				}
			}
			
		} else {
			
			for (int c = 0; c < nc; c++) {
				g += (-1) * (SUtils.ind(c, x_C) - probs[c]) * 
						dParameters_.getParameters()[c] * dParameters_.getClassProbabilities()[c];
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					g += (-1) * (SUtils.ind(c, x_C) - probs[c]) * 
							wd.getXYParameter((int)uval, c) * wd.getXYProbability((int)uval, c);
				}
			}
		}
		
		return g;
	}
	
	private void computeGradAndUpdateParameters(Instance inst, double[] probs, int x_C, double[] G) {

		if (m_DoWANBIAC) {

			for (int c = 0; c < nc; c++) {
				double g = (-1) * (SUtils.ind(c, x_C) - probs[c]) * dParameters_.getClassProbabilities()[c];

				if (m_DoRegularization) {
					g += (m_Lambda * (dParameters_.getParameters()[c] - dParameters_.getClassProbabilities()[c]));
				}

				G[c] += (g*g);

				double stepSize = m_Eta / (smoothingParameter + Math.sqrt(G[c]));
				if (stepSize == Double.POSITIVE_INFINITY) {
					stepSize = 0.0;
				}

				dParameters_.getParameters()[c] -= stepSize * g;
			}

			for (int u = 0; u < n; u++) {
				int uval = (int) inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);
					double g =  (-1) * (SUtils.ind(c, x_C) - probs[c]) * wd.getXYProbability((int)uval, c);

					if (m_DoRegularization) {
						g += (m_Lambda * (wd.getXYParameter((int)uval, c) - wd.getXYProbability((int)uval, c)));
					}

					G[posp] += (g*g);

					double stepSize = m_Eta / (smoothingParameter + Math.sqrt(G[posp]));
					if (stepSize == Double.POSITIVE_INFINITY) {
						stepSize = 0.0;
					}

					double newval = wd.getXYParameter(uval, c) - (stepSize * g);
					wd.setXYParameter(uval, c, newval);

					dParameters_.getParameters()[posp] = newval;
				}
			}

		} else {

			for (int c = 0; c < nc; c++) {
				double g = (-1) * (SUtils.ind(c, x_C) - probs[c]);

				if (m_DoRegularization) {
					g += (m_Lambda * (dParameters_.getParameters()[c] - 0));
				}

				G[c] += (g*g);

				double stepSize = m_Eta / (smoothingParameter + Math.sqrt(G[c]));
				if (stepSize == Double.POSITIVE_INFINITY) {
					stepSize = 0.0;
				}

				dParameters_.getParameters()[c] -= stepSize * g;
			}

			for (int u = 0; u < n; u++) {
				int uval = (int) inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					double g = (-1) * (SUtils.ind(c, x_C) - probs[c]);

					if (m_DoRegularization) {
						g += (m_Lambda * (wd.getXYParameter((int)uval, c) - 0));
					}

					G[posp] += (g*g);

					double stepSize = m_Eta / (smoothingParameter + Math.sqrt(G[posp]));
					if (stepSize == Double.POSITIVE_INFINITY) {
						stepSize = 0.0;
					}

					double newval = wd.getXYParameter(uval, c) - (stepSize * g);
					wd.setXYParameter(uval, c, newval);

					dParameters_.getParameters()[posp] = newval;
				}
			}
		}
		
		if (m_DoAdaptiveRegularization) {
			
			MersenneTwister rg = new MersenneTwister();
			int randSample = rg.nextInt(instancesTrain.numInstances());

			Instance instance = instancesTrain.instance(randSample);
			int x_C_V = (int) instance.classValue();
			double[] probs_V = predict(instance);
			
			SUtils.exp(probs_V);
			
			double sum = computeGradSum(instance, probs_V, x_C_V);
			double alpha = 0.001;
			//double alpha = 0.001 * m_Eta;
			//m_Lambda = Math.max(m_Lambda + alpha *sum, 0);
			m_Lambda = m_Lambda + alpha * m_Lambda * sum;
		}

	}

	private void updateParameters(Instance inst, double[] probs, int x_C, double stepSize, double[] gradients) {

		for (int c = 0; c < nc; c++) {
			dParameters_.getParameters()[c] -= stepSize * gradients[c];
		}

		for (int u = 0; u < n; u++) {
			int uval = (int) inst.value(m_Order[u]);

			wdBayesNode wd = dParameters_.getBayesNode(inst, u);

			for (int c = 0; c < nc; c++) {
				int posp = wd.getXYIndex((int)uval, c);

				double newval = wd.getXYParameter(uval, c) - (stepSize * gradients[posp]);
				wd.setXYParameter(uval, c, newval);

				dParameters_.getParameters()[posp] = newval;
			}
		}

	}

	private void updateParameters(Instance inst, double[] probs, int x_C, double[] stepSize, double[] gradients) {

		for (int c = 0; c < nc; c++) {
			dParameters_.getParameters()[c] -= stepSize[c] * gradients[c];
		}

		for (int u = 0; u < n; u++) {
			int uval = (int) inst.value(m_Order[u]);

			wdBayesNode wd = dParameters_.getBayesNode(inst, u);

			for (int c = 0; c < nc; c++) {
				int posp = wd.getXYIndex((int)uval, c);

				double newval = wd.getXYParameter(uval, c) - (stepSize[posp] * gradients[posp]);
				wd.setXYParameter(uval, c, newval);

				dParameters_.getParameters()[posp] = newval;
			}
		}

	}

	private void computeHessian(Instance inst, double[] probs, int x_C, double[] hessians) {

		if (m_DoWANBIAC) {

			double[] d =new double[nc];

			for (int c = 0;  c < nc; c++) {
				d[c] =  (1 - probs[c]) * probs[c];
			}

			for (int c = 0; c < nc; c++) {
				hessians[c] += d[c] * dParameters_.getClassProbabilities()[c];
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {

					int posp = wd.getXYIndex((int)uval, c);

					hessians[posp] += (d[c] * wd.getXYProbability((int)uval, c) * wd.getXYProbability((int)uval, c)); 
				}
			}

		} else {

			double[] d =new double[nc];

			for (int c = 0;  c < nc; c++) {
				d[c] =  (1 - probs[c]) * probs[c];
			}

			for (int c = 0; c < nc; c++) {
				hessians[c] += d[c];
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);

				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {

					int posp = wd.getXYIndex((int)uval, c);

					hessians[posp] += d[c]; 
				}
			}
		}

	}

	public double[] distributionForInstance(Instance inst) {

		double[] probs = new double[nc];

		if (m_DoWANBIAC) {

			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c] * dParameters_.getClassProbabilities()[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(m_Order[u]);

					wdBayesNode wd = dParameters_.getBayesNode(inst, u);

					probs[c] += wd.getXYParameter((int)uval, c) * wd.getXYProbability((int)uval, c);
				}
			}

		} else {

			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(m_Order[u]);

					wdBayesNode wd = dParameters_.getBayesNode(inst, u);

					probs[c] += wd.getXYParameter((int)uval, c);
				}
			}

		}

		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		return probs;
	}

	/*
	 * -------------------------------------------------------------------------------------
	 * Cross-validation functions
	 * -------------------------------------------------------------------------------------
	 */

	private ArrayList<Integer> getTrainTestIndices(int N) {

		int Nvalidation = 0;

		if (N / 10 >= 10000) {
			Nvalidation = 10000;
		} else {
			Nvalidation = (int) N / 10;
		}

		System.out.println("Creating Validation (CV) file of size: " + Nvalidation);

		MersenneTwister rg = new MersenneTwister();

		ArrayList<Integer> indexList = new ArrayList<>();

		int nvalid = 0;
		while (nvalid < Nvalidation) {
			int index = rg.nextInt(N);
			if (!indexList.contains(index)) {
				indexList.add(index);
				nvalid++;
			}
		}

		return indexList;
	}

	private Instances[] getTrainTestInstances(File sourceFile, ArrayList<Integer> indexList) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances[] instancesList = new Instances[2];

		Instances instancesTrain = new Instances(structure);
		Instances instancesTest = new Instances(structure);

		int nvalidation = indexList.size();

		int i = 0;
		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			if (indexList.contains(i)) {
				if (nvalidation % 5 == 0) {
					instancesTest.add(row);
				} else {
					instancesTrain.add(row);
				}
				nvalidation++;
			}
			i++;
		}

		instancesList[0] = instancesTrain;
		instancesList[1] = instancesTrain;

		System.out.println("-- Train Test files created for cross-validating step size -- Train = " + instancesTrain.numInstances() + ", and Test = " + instancesTest.numInstances());

		return instancesList;
	}

	private Instances[] getTrainTestInstances(File sourceFile, int N) throws FileNotFoundException, IOException {

		int Ntrain = 0;
		int Ntest = 0;

		if (N / 10 >= 10000) {
			Ntrain = 10000;
		} else {
			Ntrain = (int) N / 10;
		}

		Ntest = Ntrain / 2;

		System.out.println("Creating Train (CV) file of size: " + Ntrain);
		System.out.println("Creating Test (CV) file of size: " + Ntest);

		MersenneTwister rg = new MersenneTwister();

		ArrayList<Integer> indexListTrain = new ArrayList<>();
		ArrayList<Integer> indexListTest = new ArrayList<>();

		for (int i = 0; i < Ntrain; i++) {
			int index = rg.nextInt(N);
			if (!indexListTrain.contains(index)) {
				indexListTrain.add(index);
			}
		}
		for (int i = 0; i < Ntest; i++) {
			int index = rg.nextInt(N);
			if (!indexListTest.contains(index) && !indexListTrain.contains(index)) {
				indexListTest.add(index);
			}
		}

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances[] instancesList = new Instances[2];

		Instances instancesTrain = new Instances(structure);
		Instances instancesTest = new Instances(structure);

		int i = 0;
		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			if (indexListTrain.contains(i)) {
				instancesTrain.add(row);
			}
			if (indexListTest.contains(i)) {
				instancesTest.add(row);
			}
			i++;
		}

		instancesList[0] = instancesTrain;
		instancesList[1] = instancesTrain;

		System.out.println("-- Train Test files created for cross-validating step size --");
		return instancesList;
	}

	/*
	 * -------------------------------------------------------------------------------------
	 * Option Setters
	 * -------------------------------------------------------------------------------------
	 */

	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		m_DoSKDB = Utils.getFlag('S', options);
		m_DoDiscriminative = Utils.getFlag('D', options);

		m_DoWANBIAC = Utils.getFlag('W', options); 

		Utils.checkForRemainingOptions(options);
	}

	/*
	 * -------------------------------------------------------------------------------------
	 * Getters and Setters
	 * -------------------------------------------------------------------------------------
	 */

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

	public int getSelectedK() {
		return m_BestK_;
	}

	public int getSelectedAttributes() {
		return m_BestattIt;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public int[][] getM_Parents() {
		return m_Parents;
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

	public String getM_O() {
		return m_O;
	}

	public void setM_O(String m_O) {
		this.m_O = m_O;
	}

	public boolean isM_DoRegularization() {
		return m_DoRegularization;
	}

	public void setM_DoRegularization(boolean m_DoRegularization) {
		this.m_DoRegularization = m_DoRegularization;
	}

	public boolean isM_DoCrossvalidate() {
		return m_DoCrossvalidate;
	}

	public  void setM_DoCrossvalidate(boolean m_DoCrossvalidate) {
		this.m_DoCrossvalidate = m_DoCrossvalidate;
	}

	public double getM_Lambda() {
		return m_Lambda;
	}

	public void setM_Lambda(double m_Lambda) {
		this.m_Lambda = m_Lambda;
	}

	public double getM_Eta() {
		return m_Eta;
	}

	public void setM_Eta(double m_Eta) {
		this.m_Eta = m_Eta;
	}

	public int getM_NumIterations() {
		return m_NumIterations;
	}

	public int getM_BufferSize() {
		return m_BufferSize;
	}

	public void setM_BufferSize(int m_BufferSize) {
		this.m_BufferSize = m_BufferSize;
	}

	public void setM_NumIterations(int m_NumIterations) {
		this.m_NumIterations = m_NumIterations;
	}

	public boolean isM_DoWANBIAC() {
		return m_DoWANBIAC;
	}

	public void setM_DoWANBIAC(boolean m_DoWANBIAC) {
		this.m_DoWANBIAC = m_DoWANBIAC;
	}

	public static boolean isM_DoAdaptiveRegularization() {
		return m_DoAdaptiveRegularization;
	}

	public static void setM_DoAdaptiveRegularization(boolean m_DoAdaptiveRegularization) {
		fuplaOOC.m_DoAdaptiveRegularization = m_DoAdaptiveRegularization;
	}

}
