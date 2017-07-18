package Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.core.converters.ArffLoader.ArffReader;

public class SUtils {

	public static int minNumThreads = 4000;
	public static int displayPerfAfterInstances = 1000;
	public static String perfOutput = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-+*`~!@#$%^&_|:;'?";
	public static int m_Limit = 1;
	
	
	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static double MEsti(double freq1, double freq2, double numValues) {
		double m_MEsti = 1.0;
		double mEsti = (freq1 + m_MEsti / numValues) / (freq2 + m_MEsti);
		return mEsti;
	}

	public static double MEsti(double freq1, double freq2) {
		double mEsti = freq1 / freq2;
		return mEsti;
	}

	public static void boundAndNormalizeInLogDomain(double[] logs,
			double maxDifference) {
		boundDifferences(logs, maxDifference);
		double logSum = sumInLogDomain(logs);
		for (int i = 0; i < logs.length; i++)
			logs[i] -= logSum;
	}

	public static void boundDifferences(double[] logs, double maxDifference) {
		double maxLog = logs[0];
		for (int i = 1; i < logs.length; i++) {
			if (maxLog < logs[i]) {
				maxLog = logs[i];
			}
		}
		for (int i = 0; i < logs.length; i++) {
			logs[i] = logs[i] - maxLog;
			if (logs[i] < -maxDifference) {
				logs[i] = -maxDifference;
			}
		}
	}

	public static void normalizeInLogDomain(double[] logs) {
		double logSum = sumInLogDomain(logs);
		for (int i = 0; i < logs.length; i++)
			logs[i] -= logSum;
	}

	public static double sumInLogDomain(double[] logs) {
		// first find max log value
		double maxLog = logs[0];
		int idxMax = 0;
		for (int i = 1; i < logs.length; i++) {
			if (maxLog < logs[i]) {
				maxLog = logs[i];
				idxMax = i;
			}
		}
		// now calculate sum of exponent of differences
		double sum = 0;
		for (int i = 0; i < logs.length; i++) {
			if (i == idxMax) {
				sum++;
			} else {
				sum += Math.exp(logs[i] - maxLog);
			}
		}
		// and return log of sum
		return maxLog + Math.log(sum);
	}

	public static void exp(double[] logs) {
		for (int c = 0; c < logs.length; c++) {
			logs[c] = Math.exp(logs[c]);
		}
	}

	public static void log(double[] logs) {
		for (int c = 0; c < logs.length; c++) {
			logs[c] = Math.log(logs[c]);
		}
	}	

	public static int[] sort(double[] mi) {		
		int[] sortedPositions = Utils.sort(mi);
		int n = mi.length;
		int[] order = new int[n];
		for (int i = 0; i < n; i++) {
			order[i] = sortedPositions[(n-1) - i];
		}
		return order;
	}

	public static int combination(int N, int k) {
		int n = 0;
		int num = factorial(N);
		int denum1 = factorial(N - k);
		int denum2 = factorial(k);
		n = (num) / (denum1 * denum2);
		return n;
	}

	public static int factorial(int a) {
		int facta = 1;		
		for (int i = a; i > 0; i--) {
			facta *= i;
		}		
		return facta;
	}

	public static int NC2(int a) {
		int count = 0;
		for (int att1 = 1; att1 < a; att1++) {			
			for (int att2 = 0; att2 < att1; att2++) {
				count++;	
			}
		}
		return count;		
	}

	public static int NC3(int a) {
		int count = 0;
		for (int att1 = 2; att1 < a; att1++) {			
			for (int att2 = 1; att2 < att1; att2++) {				
				for (int att3 = 0; att3 < att2; att3++) {
					count++;
				}
			}
		}
		return count;		
	}

	public static int NC4(int a) {
		int count = 0;

		for (int att1 = 3; att1 < a; att1++) {			
			for (int att2 = 2; att2 < att1; att2++) {				
				for (int att3 = 1; att3 < att2; att3++) {
					for (int att4 = 0; att4 < att3; att4++) {
						count++;
					}
				}
			}
		}
		return count;
	}

	public static int NC5(int a) {
		int count = 0;

		for (int att1 = 4; att1 < a; att1++) {			
			for (int att2 = 3; att2 < att1; att2++) {				
				for (int att3 = 2; att3 < att2; att3++) {
					for (int att4 = 1; att4 < att3; att4++) {
						for (int att5 = 0; att5 < att4; att5++) {
							count++;
						}
					}
				}
			}
		}
		return count;
	}
	
	public final static void randomize(int[] index, int n) {
		
		Random random = new Random(System.currentTimeMillis());
		
		for (int i = 0; i < index.length; i++) {
			int k = random.nextInt(n);
			index[i] = k;
		}
		
	}

	public final static void randomize(int[] index) {
		Random random = new Random(System.currentTimeMillis());
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}
	
	public final void randomize(int[] index, Random random) {
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}
	
	public static void shuffleArray(int[] ar) {

		Random rnd = new Random();
		for (int i = ar.length - 1; i > 0; i--)
		{
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}

	public static double maxAbsValueInAnArray(double[] array) {
		int index = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (Math.abs(array[i]) > max) {
				max = Math.abs(array[i]);
				index = i;				
			}
		}		
		return Math.abs(array[index]);
	}

	public static int maxLocationInAnArray(double[] array) {
		int index = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				index = i;				
			}
		}		
		return index;
	}

	public static int minLocationInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] < min) {
				min = array[i];
				index = i;				
			}
		}		
		return index;
	}

	public static int findMaxValueLocationInNDMatrix(double[][] results, int dim) {
		double[] tempVector = new double[results.length];

		for (int i = 0; i < results.length; i++) {
			tempVector[i] = results[i][dim];
		}

		int index = minLocationInAnArray(tempVector);

		return index;
	}

	public static double MI(long[][] contingencyMatrix) {
		int n = 0;
		int nrows = contingencyMatrix.length;
		int ncols = contingencyMatrix[0].length;
		
		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += contingencyMatrix[r][c];
				colsSum[c] += contingencyMatrix[r][c];
				n += contingencyMatrix[r][c];
			}				
		}

		double MI = 0;

		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (contingencyMatrix[r][c] > 0) {
							double a = contingencyMatrix[r][c] / ( rowsSum[r]/(double)n * colsSum[c] ) ;
							MI += contingencyMatrix[r][c]/(double)n * Math.log(a / Math.log(2));
						}
					}
				}
			}
		}
		return MI;
	}
	
	public static double gsquare(long[][]observed){
		long n = 0;
		int nrows = observed.length;
		int ncols = observed[0].length;
		
		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += observed[r][c];
				colsSum[c] += observed[r][c];
				n += observed[r][c];
			}				
		}

		double gs = 0.0;
		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (observed[r][c] > 0) {
							double exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n) ;
							gs+= 1.0*observed[r][c]*FastMath.log(observed[r][c]/exp);
						}
					}
				}
			}
		}
		gs*=2.0;
		return gs;
	}
	
	public static double chisquare(long[][]observed){
		long n = 0;
		int nrows = observed.length;
		int ncols = observed[0].length;
		
		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += observed[r][c];
				colsSum[c] += observed[r][c];
				n += observed[r][c];
			}				
		}

		double chi = 0.0;
		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (observed[r][c] > 0) {
							double exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n) ;
							double diff = observed[r][c]-exp;
							chi+= diff*diff/exp;
						}
					}
				}
			}
		}
		return chi;
	}

	public static boolean inSet(int[] array, int element) {
		boolean present = false;
		
		if (array == null) {
			return false;
		}
		
		for (int i = 0; i < array.length; i++) {
			if (array[i] == element) {
				present = true;
				break;
			}
		}
		return present;
	}
	
	public static boolean linkExist(int[][] m_Parents, int[] m_Order, int i, int j) {
		boolean present = false;
		
		if (inSet(m_Parents[i], m_Order[j]))
			present = true;		
		
		if (inSet(m_Parents[j], m_Order[i]))
			present = true;
		
		return present;
	}
	
	public static int[] CheckForPerfectness(int[] m_TempParents, int[][] m_Parents, int[] m_Order) {
		int[] parents = null;		
		parents = new int[m_TempParents.length];
		
		int j = 0;
		for (int j1 = 0; j1 < m_TempParents.length; j1++) {
			for (int j2 = j1 + 1; j2 < m_TempParents.length; j2++) {
				
				if (SUtils.linkExist(m_Parents, m_Order, m_TempParents[j1], m_TempParents[j2])) {
					parents[j] = m_TempParents[j1];
					parents[j+1] = m_TempParents[j2];
					j+=2;
				}
				
			}
		}	
		
		return null;
	}

	public static void labelsToProbs(int[] labels, double[] probs) {
		int[] count = new int[probs.length];
		for (int i = 0; i < labels.length; i++) {
			count[labels[i]]++;
		}
		int maxCount = Integer.MIN_VALUE;
		int labelIndex = -1;
		for (int i = 0; i < count.length; i++) {
			if (count[i] > maxCount) {
				maxCount = count[i];
				labelIndex = i;
			}
		}
		
		probs[labelIndex] = 1.0;
	}
	
	public static boolean inArray(int val, int[] arr) {
		boolean flag = false;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == val) {
				flag = true;
			}
		}
		return flag;
	}
	
	public static double minAbsValueInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (Math.abs(array[i]) < min) {
				min = Math.abs(array[i]);
				index = i;				
			}
		}		
		return Math.abs(array[index]);
	}
	
	public static double minNonZeroValueInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] != 0 && array[i] < min) {
				min = Math.abs(array[i]);
				index = i;				
			}
		}		
		return array[index];
	}

	public static Instances generateBaggedData(Instances instances) {
		int N = instances.numInstances();
		Random random = new Random(System.currentTimeMillis());
		
		Instances data = new Instances(instances, 0);
		
		for (int i = 0; i < N; i++) {
			int index = random.nextInt(N);
			data.add(instances.instance(index));
		}
		
		return data;
	}
	
public static File discretizeData(File sourceFile, Instances CVInstances, int BUFFER_SIZE, int ARFF_BUFFER_SIZE, int type, int numBins) throws Exception {
		
		File out = File.createTempFile("trainCV-", ".arff");

		Instances m_DiscreteInstances = null;

		if (type == 1) {
			
			/* MDL based discretization */

			System.out.println("Starting MDL Discretization");
			
			weka.filters.supervised.attribute.Discretize m_Disc = null;

			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(CVInstances);

			m_DiscreteInstances = weka.filters.Filter.useFilter(CVInstances, m_Disc);

			out.deleteOnExit();

			ArffSaver fileSaver = new ArffSaver();
			fileSaver.setFile(out);
			fileSaver.setRetrieval(Saver.INCREMENTAL);
			fileSaver.setStructure(m_DiscreteInstances);

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			int i = 0;
			while ((row = reader.readInstance(structure)) != null)  {
				m_Disc.input(row);
				row = m_Disc.output();

				fileSaver.writeIncremental(row);
				i++;

			}
		} else if (type == 2) {
			
			/* Equal Frequency based discretization */
			
			System.out.println("Starting Equal Frequency Discretization with " + numBins + " bins.");
			
			weka.filters.unsupervised.attribute.Discretize m_Disc = null;

			m_Disc = new weka.filters.unsupervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(CVInstances);
			m_Disc.setBins(numBins);

			m_DiscreteInstances = weka.filters.Filter.useFilter(CVInstances, m_Disc);

			out.deleteOnExit();

			ArffSaver fileSaver = new ArffSaver();
			fileSaver.setFile(out);
			fileSaver.setRetrieval(Saver.INCREMENTAL);
			fileSaver.setStructure(m_DiscreteInstances);

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			int i = 0;
			while ((row = reader.readInstance(structure)) != null)  {
				m_Disc.input(row);
				row = m_Disc.output();

				fileSaver.writeIncremental(row);
				i++;

			}
			
		}

		return out;
	}

	public static File normalizeData(File sourceFile, Instances CVInstances, int BUFFER_SIZE, int ARFF_BUFFER_SIZE) throws Exception {

		System.out.println("Starting Normalization");

		Instances m_NormalizedInstances = null;

		weka.filters.unsupervised.attribute.Normalize m_Norm = null;

		m_Norm = new weka.filters.unsupervised.attribute.Normalize();
		m_Norm.setInputFormat(CVInstances);

		m_NormalizedInstances = weka.filters.Filter.useFilter(CVInstances, m_Norm);

		File out = File.createTempFile("trainCV-", ".arff");
		out.deleteOnExit();

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(m_NormalizedInstances);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance row;
		int i = 0;
		while ((row = reader.readInstance(structure)) != null)  {
			m_Norm.input(row);
			row = m_Norm.output();

			fileSaver.writeIncremental(row);
			i++;

		}

		return out;
	}
	
	public static BitSet getStratifiedIndices(File sourceFile, int BUFFER_SIZE, int ARFF_BUFFER_SIZE, int S) throws FileNotFoundException, IOException {

		BitSet res = new BitSet();

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), ARFF_BUFFER_SIZE);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();

		int[] classCount = new int[nc];
		int[] numToBeSelected = new int[nc];
		int[] numSelected = new int[nc];
		double[] selectionProb = new double[nc];

		Instance current;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();
			classCount[x_C]++;
		}

		for (int c = 0; c < nc; c++) {
			if (classCount[c] < 50) {
				numToBeSelected[c] = classCount[c]/2;
				selectionProb[c] = 0.5;
			} else { 
				numToBeSelected[c] = (classCount[c]/100) * S;
				selectionProb[c] = (double) S / 100;
			}
		}

		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), ARFF_BUFFER_SIZE);
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();

			if (Math.random() < selectionProb[x_C]) {
				res.set(lineNo);
				numSelected[x_C]++;
			}

			lineNo++;
		}

		System.out.println("-------------------------------------------------------------------");
		System.out.println("Class Counts = " + Arrays.toString(classCount));
		System.out.println("Num to be selected = " + Arrays.toString(numToBeSelected));
		System.out.println("Actually selected = " + Arrays.toString(numSelected));
		System.out.println("-------------------------------------------------------------------");

		return res;
	}

	public static Instances getTrainTestInstances(File sourceFile, BitSet res, int BUFFER_SIZE, int ARFF_BUFFER_SIZE) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), ARFF_BUFFER_SIZE);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances CVInstances = new Instances(structure);

		int i = 0;
		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			if (res.get(i)) {
				CVInstances.add(row);
			}
			i++;
		}

		System.out.println("-- CVInstances file created (in memory) -- Size = " + CVInstances.numInstances());

		return CVInstances;
	}


}


