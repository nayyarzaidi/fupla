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

public class fuplaOOCFP {

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

		Instance row;

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

		//System.out.println(dParameters_.getNLL_MAP(m_Instances));
		
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

	public int getSelectedK() {
		return m_BestK_;
	}

	public void setSelectedK(int m_BestK_) {
		this.m_BestK_ = m_BestK_;
	}

	public int getSelectedAttributes() {
		return m_BestattIt;
	}

	public void setSelectedAttributes(int m_BestattIt) {
		this.m_BestattIt = m_BestattIt;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public void setM_Order(int[] order) {
		m_Order = order;
	}

	public int[][] getM_Parents() {
		return m_Parents;
	}

	public void setM_Parents(int[][] parents) {
		m_Parents = parents;
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
