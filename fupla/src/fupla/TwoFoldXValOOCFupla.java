package fupla;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;

import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class TwoFoldXValOOCFupla {

	private static String data = "";

	private static boolean m_MVerb = false; 					      // -V
	private static boolean m_DoDiscriminative = false; 	      // -D

	private static int m_KDB = 1;										       // -K
	private static String m_O = "adagrad";                             // -O

	private static boolean m_DoSKDB = false;				      // -S

	private static boolean m_DoRegularization = false;		  // -R
	private static int m_DoAdaptiveRegularization = 0;		      // -A

	private static double m_Lambda = 0.001;						  // -L
	private static double m_Eta = 0.01;                                 // -E
	private static boolean m_DoCrossvalidate = false;          // -C

	private static int m_NumIterations = 1;                            // -I
	private static int m_BufferSize = 1;                                  // -B

	private static boolean m_DoWANBIAC = false;               // -W
	private static boolean m_EndIterFlag = false;               // -Z

	private static int m_nExp 						= 5;                  // -X
	private static int m_Folds 						= 2;                  // -Y

	private static boolean m_Discretization = false;          // -Q

	public static final int BUFFER_SIZE = 10*1024*1024; 	//100MB
	public static final int ARFF_BUFFER_SIZE = 100000;

	public static void main(String[] args) throws Exception {

		System.out.println("Executing TwoFoldXValOOCFupla");

		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File sourceFile;
		sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}

		/*
		 * Read file sequentially, 10000 instances at a time
		 */
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);

		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
		int N = getNumData(sourceFile, structure);
		System.out.println("Read " + N + " datapoints");

		/*
		 * Discretize data and store on the disk
		 */
		File disc_sourceFile = null;
		if (m_Discretization) {
			int S = 5; // use 5% of the data for getting these indices
			BitSet res = SUtils.getStratifiedIndices(sourceFile, BUFFER_SIZE, ARFF_BUFFER_SIZE, S);
			Instances CVInstances = SUtils.getTrainTestInstances(sourceFile, res, BUFFER_SIZE, ARFF_BUFFER_SIZE);

			disc_sourceFile = SUtils.discretizeData(sourceFile, CVInstances, BUFFER_SIZE, ARFF_BUFFER_SIZE, 2, 500);
			System.out.println("Discretized Instances created at:  '" + disc_sourceFile.getAbsolutePath() + "'");
			sourceFile = disc_sourceFile;

			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);

			structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);
			nc = structure.numClasses();
			N = getNumData(sourceFile, structure);
			System.out.println("Read " + N + " datapoints");
		}

		double m_RMSE = 0;
		double m_Error = 0;
		double m_LogLoss = 0;

		int NTest = 0;
		long seed = 3071980;

		/*
		 * Start m_nExp rounds of Experiments
		 */

		int lineNo = 0;
		Instance current;
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		while ((current = reader.readInstance(structure)) != null) {
			lineNo++;
		}

		double[][] instanceProbs = new double[lineNo][nc];

		double trainTime = 0, testTime = 0;
		double trainStart = 0, testStart = 0;

		for (int exp = 0; exp < m_nExp; exp++) {

			seed++;

			if (m_MVerb) {
				System.out.println("Experiment No. " + exp);
			}

			MersenneTwister rg = new MersenneTwister(seed);
			BitSet indexes = null;

			for (int fold = 0; fold < 2; fold++) {
				if (m_MVerb) {
					System.out.println("Fold No. " + fold);
				}	

				if (fold == 0) {

					BitSet test0Indexes = getTest0Indexes(sourceFile, structure, rg);
					indexes = test0Indexes;

				} else if (fold == 1) {

					BitSet test1Indexes = new BitSet(lineNo);
					test1Indexes.set(0, lineNo);
					test1Indexes.xor(indexes);

					indexes = test1Indexes;
				}

				//System.out.println(indexes);

				// ---------------------------------------------------------
				// Train on this fold
				// ---------------------------------------------------------

				File trainFile;

				fuplaOOC learner = new fuplaOOC();

				learner.setM_MVerb(m_MVerb);
				learner.setM_KDB(m_KDB);
				learner.setM_DoSKDB(m_DoSKDB);
				learner.setM_DoDiscriminative(m_DoDiscriminative);
				if (m_DoDiscriminative) {
					learner.setM_O(m_O);
					learner.setM_Eta(m_Eta);
					learner.setM_DoRegularization(m_DoRegularization);
					if (m_DoRegularization) {
						learner.setM_DoAdaptiveRegularization(m_DoAdaptiveRegularization);
						learner.setM_Lambda(m_Lambda);
					}
					learner.setM_NumIterations(m_NumIterations);
					learner.setM_BufferSize(m_BufferSize);   
					learner.setM_DoCrossvalidate(m_DoCrossvalidate);
				}
				learner.setM_DoWANBIAC(m_DoWANBIAC);

				// creating tempFile for train0
				trainFile = createTrainTmpFile(sourceFile, structure, indexes);
				System.out.println("Train file generated");

				if (m_MVerb) {
					System.out.println("Training fold 0: trainFile is '" + trainFile.getAbsolutePath() + "'");
				}

				trainStart = System.currentTimeMillis();

				learner.buildClassifier(trainFile);	

				trainTime += (System.currentTimeMillis() - trainStart);

				// ---------------------------------------------------------
				// Test on this fold
				// ---------------------------------------------------------
				if (m_MVerb) {
					System.out.println("Testing fold 0 started");
				}

				testStart = System.currentTimeMillis();

				int thisNTest = 0;

				lineNo = 0;
				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
				while ((current = reader.readInstance(structure)) != null) {
					if (indexes.get(lineNo)) {
						double[] probs = new double[nc];
						int x_C = (int) current.classValue();

						probs = learner.distributionForInstance(current);	

						// ------------------------------------
						// Update Error and RMSE
						// ------------------------------------
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

						m_LogLoss -= Math.log(probs[x_C]);

						thisNTest++;
						NTest++;

						instanceProbs[lineNo][pred]++;
					}
					lineNo++;
				}

				testTime += System.currentTimeMillis() - testStart;

				if (m_MVerb) {
					System.out.println("Testing fold " + fold + " finished - 0-1=" + (m_Error / NTest) + "\trmse=" + Math.sqrt(m_RMSE / NTest) + "\tlogloss=" + m_LogLoss / (nc * NTest));
				}

				if (Math.abs(thisNTest - indexes.cardinality()) > 1) {
					System.err.println("no! " + thisNTest + "\t" + indexes.cardinality());
				}

			} // Ends No. of Folds
			
		} // Ends No. of Experiments


		double m_Bias = 0;
		double m_Sigma = 0;
		double m_Variance = 0;

		lineNo = 0;
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		while ((current = reader.readInstance(structure)) != null) {
			double[] predProbs = instanceProbs[lineNo];

			double pActual, pPred;
			double bsum = 0, vsum = 0, ssum = 0;
			for (int j = 0; j < nc; j++) {
				pActual = (current.classValue() == j) ? 1 : 0;
				pPred = predProbs[j] / m_nExp;
				bsum += (pActual - pPred) * (pActual - pPred) - pPred * (1 - pPred) / (m_nExp - 1);
				vsum += (pPred * pPred);
				ssum += pActual * pActual;
			}
			m_Bias += bsum;
			m_Variance += (1 - vsum);
			m_Sigma += (1 - ssum);

			lineNo++;
		}

		m_Bias = m_Bias / (2 * lineNo);
		m_Variance = (m_Error / NTest) - m_Bias;

		System.out.print("\nBias-Variance Decomposition\n");
		System.out.print("\nClassifier	   : FuplaOOC (K = " + m_KDB + ")");
		System.out.print( "\nData File   : " + data);
		System.out.print("\nError               : " + Utils.doubleToString(m_Error / NTest, 6, 4));
		System.out.print("\nBias^2             : " + Utils.doubleToString(m_Bias, 6, 4));
		System.out.print("\nVariance          : " + Utils.doubleToString(m_Variance, 6, 4));
		System.out.print("\nRMSE              : " + Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4));
		System.out.print("\nLogLoss          : " + Utils.doubleToString(m_LogLoss / (nc * NTest), 6, 4));
		System.out.print("\nTraining Time  : " + Utils.doubleToString(trainTime/1000, 6, 4));
		System.out.print("\nTesting Time   : " + Utils.doubleToString(testTime/1000, 6, 4));
		System.out.print("\n\n\n");

	}

	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static void setOptions(String[] options) throws Exception {

		String Strain = Utils.getOption('t', options);
		if (Strain.length() != 0) {
			data = Strain;
		}

		m_MVerb = Utils.getFlag('V', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		m_DoSKDB = Utils.getFlag('S', options);
		m_DoDiscriminative = Utils.getFlag('D', options);

		String strX = Utils.getOption('X', options);
		if (strX.length() != 0) {
			m_nExp = Integer.valueOf(strX);
		}

		String Soutput = Utils.getOption('O', options);
		if (Soutput.length() != 0) {
			m_O = Soutput;
		}

		String strI = Utils.getOption('I', options);
		if (strI.length() != 0) {
			m_NumIterations = Integer.valueOf(strI);
		}

		String strB = Utils.getOption('B', options);
		if (strB.length() != 0) {
			m_BufferSize = Integer.valueOf(strB);
		}

		m_DoRegularization = Utils.getFlag('R', options);

		if (m_DoRegularization) {
			//m_DoAdaptiveRegularization = Utils.getFlag('A', options);

			String strA= Utils.getOption('A', options);
			if (strA.length() != 0) {
				m_DoAdaptiveRegularization = Integer.valueOf(strA);
			}

			String strL = Utils.getOption('L', options);
			if (strL.length() != 0) {
				m_Lambda = Double.valueOf(strL);
			}
		}

		String strE = Utils.getOption('E', options);
		if (strE.length() != 0) {
			m_Eta = Double.valueOf(strE);
		}

		m_DoCrossvalidate =  Utils.getFlag('C', options);

		m_DoWANBIAC = Utils.getFlag('W', options); 

		Utils.checkForRemainingOptions(options);

	}

	private static int getNumData(File sourceFile, Instances structure) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			if(nLines%1000000==0){
				System.out.println(nLines);
			}
			nLines++;
		}
		return nLines;
	}

	private static BitSet getTest0Indexes(File sourceFile, Instances structure, MersenneTwister rg) throws FileNotFoundException, IOException {
		BitSet res = new BitSet();
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			if (rg.nextBoolean()) {
				res.set(nLines);
			}
			nLines++;
		}

		int expectedNLines = (nLines % 2 == 0) ? nLines / 2 : nLines / 2 + 1;
		int actualNLines = res.cardinality();

		if (actualNLines < expectedNLines) {
			while (actualNLines < expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (res.get(chosen));
				res.set(chosen);
				actualNLines++;
			}
		} else if (actualNLines > expectedNLines) {
			while (actualNLines > expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (!res.get(chosen));
				res.clear(chosen);
				actualNLines--;
			}
		}
		return res;
	}

	public static File createTrainTmpFile(File sourceFile, Instances structure, BitSet testIndexes) throws IOException {
		File out = File.createTempFile("train-", ".arff");
		out.deleteOnExit();
		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);

		Instance current;
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			if (!testIndexes.get(lineNo)) {
				fileSaver.writeIncremental(current);
			}
			lineNo++;
		}
		fileSaver.writeIncremental(null);
		return out;
	}

	public static String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

}
