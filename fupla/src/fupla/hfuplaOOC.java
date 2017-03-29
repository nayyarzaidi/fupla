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
import org.apache.commons.math3.random.RandomGenerator;

public class hfuplaOOC {

	private Instances structure;

	int N;
	int n;
	int nc;
	public int[] paramsPerAtt;

	public xxyDist xxyDist_;
	public wdBayesHierarchicalParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private boolean m_MVerb = false; 							// -V
	private boolean m_DoSKDB = false;						// -S
	private boolean m_DoDiscriminative = false;			// -D
	private int m_KDB = 1; 											// -K
	private String m_O = "adagrad";								// -O

	private boolean m_DoWANBIAC = false;               // -W

	int m_BestK_ = 0; 
	int m_BestattIt = 0;

	private RandomGenerator rg = null;
	private static final int BUFFER_SIZE = 100000;

	private static int maxIterations = 1;						// -I 
	
	public void buildClassifier(File sourceFile) throws Exception {

		System.out.println("[----- hfuplaOOC -----]: Reading structure -- " + sourceFile);

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

		dParameters_ = new wdBayesHierarchicalParametersTree(n, nc, paramsPerAtt, m_Order, m_Parents, 1);

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
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			/* Start the book keeping, select the best k and best attributes */
			//			for (int k = 0; k <= m_KDB; k++) {
			//				System.out.println("k = " + k);
			//				for (int u = 0; u < n; u++){
			//					System.out.print(foldLossFunctallK_[k][u] + ", ");
			//				}
			//				System.out.println(foldLossFunctallK_[k][n]);
			//			}

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

			int np = dParameters_.getNp();

			Instances instancesTrain = null;
			Instances instancesTest = null;

			Instances[] instanceList;
			//instanceList = getTrainTestInstances(sourceFile, N);

			ArrayList<Integer> indexList = getTrainTestIndices(N);
			instanceList = getTrainTestInstances(sourceFile, indexList);

			Collections.sort(indexList);

			instancesTrain = instanceList[0];
			instancesTest = instanceList[1];

			if (m_O.equalsIgnoreCase("sgd")) {

				// alpha = learned from the data

				System.out.println("Finding Alpha (sgd), Please Wait");
				double alpha = optimizeAlphaSGD(sourceFile, instancesTrain, instancesTest);
				System.out.println("Using alpha = " + alpha);

				for (int i = 0; i < maxIterations; i++) {
					System.out.println("Iteration: " + i);

					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
					this.structure = reader.getStructure();
					structure.setClassIndex(structure.numAttributes() - 1);

					int numdata = 0;

					while ((row = reader.readInstance(structure)) != null)  {

						if (Collections.binarySearch(indexList, numdata) >= 0) {
							continue;
						}

						double[] gradients = new double[np];

						int x_C = (int) row.classValue();
						double[] probs = predict(row);
						SUtils.exp(probs);

						computeGrad(row, probs, x_C, gradients);

						// Updated parameters
						dParameters_.updateParameters(alpha, gradients);

						numdata++;

						if (numdata % 1000 == 0) {
							System.out.println(numdata);
						}
					}
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

			} else if (m_O.equalsIgnoreCase("adagrad")) {

				// \alpha = \alpha_0 / \sum_ grad
				// \alpha_0 is learned from the data

				double smoothingParameter = 1e-9;

				System.out.println("Finding Alpha (adagrad), Please Wait");
				//double alpha = optimizeAlphaADAGRAD(sourceFile, instancesTrain, instancesTest, smoothingParameter);
				double alpha = 0.0001;
				System.out.println("Using alpha = " + alpha);

				double[] G = new double[np];

				for (int iter = 0; iter < maxIterations; iter++) {
					System.out.println("Iteration: " + iter);

					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
					this.structure = reader.getStructure();
					structure.setClassIndex(structure.numAttributes() - 1);

					int numdata = 0;

					while ((row = reader.readInstance(structure)) != null)  {

						if (Collections.binarySearch(indexList, numdata) >= 0) {
							continue;
						}

						double[] gradients = new double[np];

						int x_C = (int) row.classValue();
						double[] probs = predict(row);
						SUtils.exp(probs);

						computeGrad(row, probs, x_C, gradients);

						for (int j = 0; j < np; j++) {
							G[j] += ((gradients[j] * gradients[j]));
						}

						double stepSize[] = new double[np];
						for (int j = 0; j < np; j++) {
							stepSize[j] = alpha / (smoothingParameter + Math.sqrt(G[j]));

							if (stepSize[j] == Double.POSITIVE_INFINITY) {
								stepSize[j] = 0.0;
							}
						}

						// Updated parameters
						dParameters_.updateParameters(stepSize, gradients);

						numdata++;

						if (numdata % 1000 == 0) {
							System.out.println(numdata);
						}
					}
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

			} else if (m_O.equalsIgnoreCase("bottousgd")) {

			} else if (m_O.equalsIgnoreCase("plr")) {

				for (int iter = 0; iter < maxIterations; iter++) {
					System.out.println("Iteration: " + iter);

					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
					this.structure = reader.getStructure();
					structure.setClassIndex(structure.numAttributes() - 1);

					int numdata = 0;

					double epsilon = 1e-9;

					double[] gbar = new double[np];
					double[] vbar = new double[np];
					double[] hbar = new double[np];
					double[] vpart = new double[np];
					double[] taus = new double[np];

					for (int i = 0; i < np; i++) {
						gbar[i] = 0;
						vbar[i] = 1.0 * epsilon;
						hbar[i] = 1.0;

						vpart[i] = (gbar[i] * gbar[i]) / vbar[i];
						taus[i] = (1.0 + epsilon) * 2;
					}

					while ((row = reader.readInstance(structure)) != null)  {

						double[] gradients = new double[np];
						double[] hessians = new double[np];

						int x_C = (int) row.classValue();
						double[] probs = predict(row);
						SUtils.exp(probs);

						computeGrad(row, probs, x_C, gradients);
						computeHessian(row, probs, x_C, hessians);						

						for (int j = 0; j < np; j++) {
							gbar[j] = (1 - 1/taus[j]) * gbar[j] + 1/taus[j] * gradients[j];
							vbar[j] = (1 - 1/taus[j]) * vbar[j] + 1/taus[j] * gradients[j] * gradients[j];
							hbar[j] = (1 - 1/taus[j]) * hbar[j] + 1/taus[j] * hessians[j];

							vpart[j] = 0;
							vpart[j] = vpart[j] + (gbar[j] * gbar[j]) / vbar[j];

							taus[j] = (1 - vpart[j]) * taus[j];
							taus[j] += (1 + epsilon);
						}

						double stepSize[] = new double[np];
						for (int j = 0; j < np; j++) {
							stepSize[j] = vpart[j] / (hbar[j] + epsilon);
						}

						// Updated parameters
						dParameters_.updateParameters(stepSize, gradients);

						numdata++;

						if (numdata % 1000 == 0) {
							System.out.println(numdata);
						}
					}
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
					// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
				}

			}

		}

		System.out.println("Finish training");
	}

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

	private double optimizeAlphaADAGRAD(File sourceFile, Instances instancesTrain, Instances instancesTest, double smoothingParameter) {

		Instance row;
		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			System.out.print(".");

			dParameters_.initializeParametersWithVal(1.0);
			double[] G = new double[np];

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);
				double[] gradients = new double[np];

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

		dParameters_.initializeParametersWithVal(1);

		return alpha[SUtils.minLocationInAnArray(perf)];
	}

	private double optimizeAlphaSGD(File sourceFile, Instances instancesTrain, Instances instancesTest) {

		int np = dParameters_.getNp();

		double[] alpha = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
		double[] perf = new double[alpha.length]; 

		for (int i = 0; i < alpha.length; i++) {

			System.out.print(".");

			dParameters_.initializeParametersWithVal(1.0);

			/* Train Classifier  with alpha i */
			for (int ii = 0; ii < instancesTrain.numInstances(); ii++) {
				Instance instance = instancesTrain.instance(ii);
				double[] gradients = new double[np];

				int x_C = (int) instance.classValue();
				double[] probs = predict(instance);
				SUtils.exp(probs);

				computeGrad(instance, probs, x_C, gradients);

				dParameters_.updateParameters(alpha[i], gradients);
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

		dParameters_.initializeParametersWithVal(1);

		return alpha[SUtils.minLocationInAnArray(perf)];
	}

	
	private double[] predict(Instance inst) {
		double[] probs = new double[nc];
		
		if (m_DoWANBIAC) {
			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c] * dParameters_.getClassProbabilities()[c];

				for (int u = 0; u < n; u++) {
					wdBayesNode pt = dParameters_.getBayesNode(u);
					dParameters_.resetProb();
					int uval = (int)inst.value(m_Order[u]);
					probs[c] += dParameters_.predict(inst, u, uval, c, pt);
				}
			}
		} else {
			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c];

				for (int u = 0; u < n; u++) {
					wdBayesNode pt = dParameters_.getBayesNode(u);
					dParameters_.resetProb();
					int uval = (int)inst.value(m_Order[u]);
					probs[c] += dParameters_.predict(inst, u, uval, c, pt);
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
				wdBayesNode pt = dParameters_.getBayesNode(u);
				int uval = (int)inst.value(m_Order[u]);
				
				for (int c = 0; c < nc; c++) {
					dParameters_.gradients(inst, u, uval, c, pt, gradients, probs);
				}
			}

		} else {

			for (int c = 0; c < nc; c++) {
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * dParameters_.getClassProbabilities()[c];
			}

			for (int u = 0; u < n; u++) {
				wdBayesNode pt = dParameters_.getBayesNode(u);
				int uval = (int)inst.value(m_Order[u]);
				
				for (int c = 0; c < nc; c++) {
					dParameters_.gradients(inst, u, uval, c, pt, gradients, probs);
				}
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
					wdBayesNode pt = dParameters_.getBayesNode(u);
					dParameters_.resetProb();
					int uval = (int)inst.value(m_Order[u]);
					probs[c] += dParameters_.predict(inst, u, uval, c, pt);
				}
			}
		} else {
			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c];

				for (int u = 0; u < n; u++) {
					wdBayesNode pt = dParameters_.getBayesNode(u);
					dParameters_.resetProb();
					int uval = (int)inst.value(m_Order[u]);
					probs[c] += dParameters_.predict(inst, u, uval, c, pt);
				}
			}
		}
		
		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		
		return probs;
	}

	public int getSelectedK() {
		return m_BestK_;
	}

	public int getSelectedAttributes() {
		return m_BestattIt;
	}

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

	public int getNInstances() {
		return N;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public wdBayesHierarchicalParametersTree getdParameters_() {
		return dParameters_;
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

	public void setRandomGenerator(RandomGenerator rg) {
		this.rg = rg;
	}

	public String getM_O() {
		return m_O;
	}

	public void setM_O(String m_O) {
		this.m_O = m_O;
	}

	public boolean isM_DoWANBIAC() {
		return m_DoWANBIAC;
	}

	public void setM_DoWANBIAC(boolean m_DoWANBIAC) {
		this.m_DoWANBIAC = m_DoWANBIAC;
	}


}
