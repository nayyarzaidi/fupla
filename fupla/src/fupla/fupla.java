package fupla;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;
import optimize.Minimizer;
import optimize.DifferentiableFunction;
import optimize.FunctionValues;

import optimize.LBFGSBException;
import optimize.Result;
import optimize.StopConditions;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

public class fupla extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances = null;

	int N;
	int n;
	int nc;
	public int[] paramsPerAtt;

	public xxyDist xxyDist_;
	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private boolean m_MVerb = false; 							 // -V
	private int m_KDB = 1; 											 // -K

	private boolean m_DoSKDB = false;                      // -S
	private boolean m_DoDiscriminative = false;         // -D

	private boolean m_MultiThreaded = false;             // -M

	private double[] probs;

	int m_BestK_ = 0; 
	int m_BestattIt = 0;

	ObjectiveFunction function_to_optimize;

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		m_Instances = new Instances(instances);

		// remove instances with missing class
		n = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();
		N = m_Instances.numInstances();

		probs = new double[nc];

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
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

		xxyDist_ = new xxyDist(m_Instances);
		xxyDist_.addToCount(m_Instances);

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

		// Update dTree_ based on parents
		for (int i = 0; i < N; i++) {
			Instance instance = m_Instances.instance(i);
			dParameters_.update(instance);
		}

		dParameters_.countsToProbability();

		System.out.println(dParameters_.getNLL_MAP(m_Instances));

		if (m_DoSKDB) {

			/*
			 * ------------------------------------------------------
			 * Pass No. 3 (SKDB)
			 * ------------------------------------------------------
			 */

			double[][] foldLossFunctallK_ = new double[m_KDB + 1][n + 1];
			double[][] posteriorDist = new double[m_KDB + 1][nc];

			/* Start the third costly pass through the data */
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
		dParameters_.initializeParametersWithVal(1.0);


		/* 
		 * ------------------------------------------------------
		 * Pass No. 4 (SGD)
		 * ------------------------------------------------------
		 */

		if (m_DoDiscriminative) {

			if (m_MultiThreaded) {
				function_to_optimize = new ParallelObjectiveFunction();
			} else {
				function_to_optimize = new ObjectiveFunction();
			}

			double maxGradientNorm = 0.000000000000000000000000000000001;
			int m_MaxIterations = 10000;

			Minimizer alg = new Minimizer();
			StopConditions sc = alg.getStopConditions();
			sc.setMaxGradientNorm(maxGradientNorm);
			sc.setMaxIterations(m_MaxIterations);
			Result result;

			System.out.println();
			System.out.print("fx_QN_" + " = [");
			alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});
			result = alg.run(function_to_optimize, dParameters_.getParameters());
			System.out.println("];");
			//System.out.println(result);
			System.out.println("NoIter = " + result.iterationsInfo.iterations); System.out.println();
			
			function_to_optimize.finish();
		}

	}

	@Override
	public double[] distributionForInstance(Instance inst) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getParameters()[c] * 
					dParameters_.getClassProbabilities()[c];

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);
				
				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				probs[c] += wd.getXYParameter((int)uval, c) *
						wd.getXYProbability((int)uval, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		return probs;
	}

	class ObjectiveFunction implements DifferentiableFunction {

		@Override
		public FunctionValues getValues(double[] params) {

			boolean regularization = false;

			double mLogNC = -Math.log(nc);
			double f = 0.0;

			int np = dParameters_.getNp();
			dParameters_.copyParameters(params);

			double[] gradients = new double[np];

			for (int i = 0; i < m_Instances.numInstances(); i++) {
				Instance inst = m_Instances.instance(i);
				int x_C = (int) inst.classValue();
				double[] probs = predict(inst);
				f += (mLogNC - probs[x_C]);
				SUtils.exp(probs);

				computeGrad(inst, probs, x_C, gradients);
			}

			if (regularization) {
				f += regularizeFunction();
				regularizeGradient(gradients);
			}

			//System.out.println(negLogLikelihood);
			return new FunctionValues(f, gradients);
		}

		private double[] predict(Instance inst) {
			double[] probs = new double[nc];

			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c] * 
						dParameters_.getClassProbabilities()[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(m_Order[u]);
					
					wdBayesNode wd = dParameters_.getBayesNode(inst, u);
					probs[c] += wd.getXYParameter((int)uval, c) * wd.getXYProbability((int)uval, c);

					//probs[c] += dParameters_.getBayesNode(inst, u).getXYParameter((int)uval, c) *
					//		dParameters_.getBayesNode(inst, u).getXYProbability((int)uval, c);

				}
			}

			SUtils.normalizeInLogDomain(probs);
			return probs;
		}

		private void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {
			for (int c = 0; c < nc; c++) {
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * 
						dParameters_.getClassProbabilities()[c];
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);
				
				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					gradients[posp] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * 
							wd.getXYProbability((int)uval, c);
				}

			}
		}

		private double regularizeFunction() {
			// TODO Auto-generated method stub
			return 0;
		}

		private void regularizeGradient(double[] gradients) {
			// TODO Auto-generated method stub
		}

		@Override
		public void finish() {
			// TODO Auto-generated method stub
			
		}

	};
	
	class ParallelObjectiveFunction extends ObjectiveFunction {
		
		private static final int minNPerThread = 10000;
		
		wdBayesNode[][] nodes;
		int nThreads;
		double[][] gs;
		private double[][] tmpProbs;
		private ExecutorService executor;
		int np;
		
		public ParallelObjectiveFunction() {
			
			np = dParameters_.getNp(); 
			
			if (N < minNPerThread) {
				nThreads=1;
			} else {
				this.nThreads = Runtime.getRuntime().availableProcessors();
				if (N/this.nThreads < minNPerThread) {
					this.nThreads = N/minNPerThread + 1;
				}
			}

			this.nodes = new wdBayesNode[nThreads][n];
			this.gs = new double[nThreads][np];
			this.tmpProbs = new double[nThreads][nc];
			executor = Executors.newFixedThreadPool(nThreads);
			
			System.out.println("Will be launching: " + nThreads + " Threads.");
		}

		@Override
		public FunctionValues getValues(double[] params) {

			double f = 0.0;
			
			dParameters_.copyParameters(params);
			double gradients[] = new double[np];

			Future<Double>[] futures = new Future[nThreads];

			int assigned = 0;
			int remaining = N;
			
			for (int th = 0; th < nThreads; th++) {
				/*
				 * Compute the start and stop indexes for thread th
				 */
				int start = assigned;
				int nInstances4Thread = remaining / (nThreads - th);
				assigned += nInstances4Thread;
				int stop = assigned-1;
				remaining -= nInstances4Thread;

				/*
				 * Calling thread
				 */
				Callable<Double> thread = new CallableCLL_w(m_Instances, start, stop, nc, nodes[th], tmpProbs[th], gs[th], dParameters_, m_Order);
				futures[th] = executor.submit(thread);
			}
			
			for (int th = 0; th < nThreads; th++) {
				try {
					f += futures[th].get();

				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (ExecutionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				for (int i = 0; i < gradients.length; i++) {
					gradients[i] += gs[th][i];
				}
			}
			
			 return new FunctionValues(f, gradients);
		}

		@Override
		public void finish(){
			executor.shutdown();
		}
	};
	
	public class CallableCLL_w implements Callable<Double>{
		
		private Instances instances;
		private int start;
		private int stop;
		wdBayesNode[] myNodes;
		private double[] myProbs;
		private wdBayesParametersTree dParameters;
		private int[] order;
		private int nc;
		private double[] g;
		private double mLogNC;
		
		public CallableCLL_w(Instances m_Instances, int start, int stop, int nc, wdBayesNode[] nodes,
				double[] myProbs, double[] g, wdBayesParametersTree dParameters_, int[] order) {
			
			this.instances = instances;
			this.start = start;
			this.stop = stop;
			this.nc= nc;
			this.myNodes = nodes;
			this.myProbs = myProbs;
			this.g = g;
			this.dParameters = dParameters;
			this.order = order;
			this.mLogNC = -Math.log(nc); 	
			
		}

		@Override
		public Double call() throws Exception {
			double f = 0.0;
			Arrays.fill(g, 0.0);
			
			for (int i = start; i <= stop; i++) {
				Instance inst = m_Instances.instance(i);
				int x_C = (int) inst.classValue();
				double[] probs = predict(inst);
				f += (mLogNC - probs[x_C]);
				SUtils.exp(probs);

				computeGrad(inst, probs, x_C, g);
			}
			
			return f;
		}
		
		private double[] predict(Instance inst) {
			double[] probs = new double[nc];

			for (int c = 0; c < nc; c++) {
				probs[c] = dParameters_.getParameters()[c] * 
						dParameters_.getClassProbabilities()[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(m_Order[u]);
					
					wdBayesNode wd = dParameters_.getBayesNode(inst, u);

					probs[c] += wd.getXYParameter((int)uval, c) *
							wd.getXYProbability((int)uval, c);

				}
			}

			SUtils.normalizeInLogDomain(probs);
			return probs;
		}

		private void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {
			for (int c = 0; c < nc; c++) {
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * 
						dParameters_.getClassProbabilities()[c];
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(m_Order[u]);
				
				wdBayesNode wd = dParameters_.getBayesNode(inst, u);

				for (int c = 0; c < nc; c++) {
					int posp = wd.getXYIndex((int)uval, c);

					gradients[posp] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * 
							wd.getXYProbability((int)uval, c);
				}

			}
		}
		
	}

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

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		m_DoSKDB = Utils.getFlag('S', options);
		m_DoDiscriminative = Utils.getFlag('D', options);

		m_MultiThreaded = Utils.getFlag('M', options); 

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

	public int getnAttributes() {
		return n;
	}

}
