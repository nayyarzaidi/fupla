package fupla;

import weka.core.Instance;
import weka.core.Instances;

public class wdBayesParametersTree {
	
	private double[] parameters;
	private int np;

	private wdBayesNode[] wdBayesNode_;
	private int[] activeNumNodes;

	private int N;
	private int n;
	private int nc;

	private int[] m_ParamsPerAtt;

	private int[] order;
	private int[][] parents;

	private double[] classCounts;
	private double[] classProbabilities;

	/**
	 * Constructor called by wdBayes
	 */
	public wdBayesParametersTree(int n, int nc, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents, int m_P) {
		this.n = n;
		this.nc = nc;

		m_ParamsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			m_ParamsPerAtt[u] = paramsPerAtt[u];
		}

		order = new int[n];
		parents = new int[n][];

		for (int u = 0; u < n; u++) {
			order[u] = m_Order[u];
		}
		
		activeNumNodes = new int[n];	

		for (int u = 0; u < n; u++) {
			if (m_Parents[u] != null) {
				parents[u] = new int[m_Parents[u].length];
				for (int p = 0; p < m_Parents[u].length; p++) {
					parents[u][p] = m_Parents[u][p];
				}
			}
		}

		wdBayesNode_ = new wdBayesNode[n];
		for (int u = 0; u < n; u++) {
			wdBayesNode_[u] = new wdBayesNode();
			wdBayesNode_[u].init(nc, paramsPerAtt[m_Order[u]]);
		}

		classCounts = new double[nc];
		classProbabilities = new double[nc];
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Update count statistics that is:  relevant ***xyCount*** in every node
	 * -----------------------------------------------------------------------------------------
	 */

	public void unUpdate(Instance instance) {
		classCounts[(int) instance.classValue()]--;

		for (int u = 0; u < n; u++) {
			unUpdateAttributeTrie(instance, u, order[u], parents[u]);
		}

		N--;
	}

	public void unUpdateAttributeTrie(Instance instance, int i, int u, int[] lparents) {

		int x_C = (int) instance.classValue();
		int x_u = (int) instance.value(u);		

		wdBayesNode_[i].decrementXYCount(x_u, x_C);	

		if (lparents != null) {

			wdBayesNode currentdtNode_ = wdBayesNode_[i];

			for (int d = 0; d < lparents.length; d++) {
				int p = lparents[d];				

				int x_up = (int) instance.value(p);

				currentdtNode_.children[x_up].decrementXYCount(x_u, x_C);
				currentdtNode_ = currentdtNode_.children[x_up];
			}
		}
	}

	public void update(Instance instance) {
		classCounts[(int) instance.classValue()]++;

		for (int u = 0; u < n; u++) {
			updateAttributeTrie(instance, u, order[u], parents[u]);
		}

		N++;
	}

	public void updateAttributeTrie(Instance instance, int i, int u, int[] lparents) {

		int x_C = (int) instance.classValue();
		int x_u = (int) instance.value(u);		

		wdBayesNode_[i].incrementXYCount(x_u, x_C);	

		if (lparents != null) {

			wdBayesNode currentdtNode_ = wdBayesNode_[i];

			for (int d = 0; d < lparents.length; d++) {
				int p = lparents[d];

				if (currentdtNode_.att == -1 || currentdtNode_.children == null) {
					currentdtNode_.children = new wdBayesNode[m_ParamsPerAtt[p]];
					currentdtNode_.att = p;	
				}

				int x_up = (int) instance.value(p);
				currentdtNode_.att = p;

				// the child has not yet been allocated, so allocate it
				if (currentdtNode_.children[x_up] == null) {
					currentdtNode_.children[x_up] = new wdBayesNode();
					currentdtNode_.children[x_up].init(nc, m_ParamsPerAtt[u]);
				} 

				currentdtNode_.children[x_up].incrementXYCount(x_u, x_C);
				currentdtNode_ = currentdtNode_.children[x_up];
			}
		}
	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Convert count into (NB) probabilities
	 * -----------------------------------------------------------------------------------------
	 */

	public void countsToProbability() {
		for (int c = 0; c < nc; c++) {
			classProbabilities[c] = Math.log(SUtils.MEsti(classCounts[c], N, nc));
		}
		for (int u = 0; u < n; u++) {
			convertCountToProbs(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void convertCountToProbs(int u, int[] lparents, wdBayesNode pt) {

		int att = pt.att;

		if (att == -1) {
			for (int y = 0; y < nc; y++) {

				int denom = 0;
				for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
					denom += pt.getXYCount(dval, y);
				}

				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double prob = Math.log(Math.max(SUtils.MEsti(pt.getXYCount(uval, y), denom, m_ParamsPerAtt[u]),1e-75));
					pt.setXYProbability(uval, y, prob);
				}
			}			
			return;
		}

		while (att != -1) {
			/*
			 * Now convert non-leaf node counts to probs
			 */
			for (int y = 0; y < nc; y++) {

				int denom = 0;
				for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
					denom += pt.getXYCount(dval, y);
				}

				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					double prob = Math.log(Math.max(SUtils.MEsti(pt.getXYCount(uval, y), denom, m_ParamsPerAtt[u]),1e-75));
					pt.setXYProbability(uval, y, prob);
				}
			}

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null) 					
					convertCountToProbs(u, lparents, next);

				// Flag end of nodes
				att = -1;				
			}			
		}

		return;
	}

	//	public void convertCountToProbs(int u, int[] lparents, wdBayesNode pt) {
	//
	//		int att = pt.att;
	//
	//		if (att == -1) {
	//			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
	//			for (int y = 0; y < nc; y++) {
	//				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
	//					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
	//				}
	//			}
	//			for (int y = 0; y < nc; y++) {
	//				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
	//					int denom = 0;
	//					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
	//						denom += tempArray[dval][y];
	//					}
	//					double prob = Math.log(Math.max(SUtils.MEsti(tempArray[uval][y], denom, m_ParamsPerAtt[u]),1e-75));
	//					pt.setXYProbability(uval, y, prob);
	//				}
	//			}			
	//			return;
	//		}
	//
	//		while (att != -1) {
	//			/*
	//			 * Now convert non-leaf node counts to probs
	//			 */
	//			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
	//			for (int y = 0; y < nc; y++) {
	//				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
	//					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
	//				}
	//			}
	//			for (int y = 0; y < nc; y++) {
	//				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
	//					int denom = 0;
	//					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
	//						denom += tempArray[dval][y];
	//					}
	//					double prob = Math.log(Math.max(SUtils.MEsti(tempArray[uval][y], denom, m_ParamsPerAtt[u]),1e-75));
	//					pt.setXYProbability(uval, y, prob);
	//				}
	//			}
	//
	//			int numChildren = pt.children.length;
	//			for (int c = 0; c < numChildren; c++) {
	//				wdBayesNode next = pt.children[c];
	//				if (next != null) 					
	//					convertCountToProbs(u, lparents, next);
	//
	//				// Flag end of nodes
	//				att = -1;				
	//			}			
	//		}
	//
	//		return;
	//	}

	//probability when using leave one out cross validation, the t value is discounted
	public double ploocv(int y, int x_C) {
		if (y == x_C)
			return SUtils.MEsti(classCounts[y] - 1, N - 1, nc);
		else
			return SUtils.MEsti(classCounts[y], N - 1, nc);
	}

	public void updateClassDistributionloocv(double[][] classDist, int i, int u, Instance instance, int m_KDB) {

		int x_C = (int) instance.classValue();
		int uval = (int) instance.value(u);

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		int depth = 0;
		while ( (att != -1)) { //We want to consider kdb k=k

			// sum over all values of the Attribute for the class to obtain count[y, parents]
			for (int y = 0; y < nc; y++) {
				int totalCount = (int) pt.getXYCount(0, y);
				for (int val = 1; val < m_ParamsPerAtt[u]; val++) {
					totalCount += pt.getXYCount(val, y);
				}    

				if (y != x_C)
					classDist[depth][y] *= SUtils.MEsti(pt.getXYCount(uval, y), totalCount, m_ParamsPerAtt[u]);
				else
					classDist[depth][y] *= SUtils.MEsti(pt.getXYCount(uval, y)-1, totalCount-1, m_ParamsPerAtt[u]);
			}

			int v = (int) instance.value(att);

			wdBayesNode next = pt.children[v];
			if (next == null) {
				for (int k = depth + 1; k <= m_KDB; k++) {
					for (int y = 0; y < nc; y++) 
						classDist[k][y] = classDist[depth][y];
				}
				return;
			};

			// check that the next node has enough examples for this value;
			int cnt = 0;
			for (int y = 0; y < nc; y++) {
				cnt += next.getXYCount(uval, y);
			}

			//In loocv, we consider minCount=1(+1), since we have to leave out i.
			if (cnt < 2) { 
				depth++;
				// sum over all values of the Attribute for the class to obtain count[y, parents]
				for (int y = 0; y < nc; y++) {
					int totalCount = (int) pt.getXYCount(0, y);
					for (int val = 1; val < m_ParamsPerAtt[u]; val++) {
						totalCount += pt.getXYCount(val, y);
					}    

					if (y != x_C)
						classDist[depth][y] *= SUtils.MEsti(pt.getXYCount(uval, y), totalCount, m_ParamsPerAtt[u]);
					else
						classDist[depth][y] *= SUtils.MEsti(pt.getXYCount(uval, y)-1, totalCount-1, m_ParamsPerAtt[u]);
				}

				for (int k = depth + 1; k <= m_KDB; k++){
					for (int y = 0; y < nc; y++) 
						classDist[k][y] = classDist[depth][y];
				}
				return;
			}

			pt = next;
			att = pt.att; 
			depth++;
		}

		// sum over all values of the Attribute for the class to obtain count[y, parents]
		for (int y = 0; y < nc; y++) {
			int totalCount = (int) pt.getXYCount(0, y);
			for (int val = 1; val < m_ParamsPerAtt[u]; val++) {
				totalCount += pt.getXYCount(val, y);
			}    
			if (y != x_C)
				classDist[depth][y] *=  SUtils.MEsti(pt.getXYCount(uval, y), totalCount, m_ParamsPerAtt[u]);
			else
				classDist[depth][y] *=  SUtils.MEsti(pt.getXYCount(uval, y)-1, totalCount-1, m_ParamsPerAtt[u]);
		}

		for (int k = depth + 1; k <= m_KDB; k++){
			for (int y = 0; y < nc; y++) 
				classDist[k][y] = classDist[depth][y];
		}

	}

	public void updateClassDistributionloocv2(double[][] posteriorDist, int i, int u, Instance instance, int m_KDB) {

		int x_C = (int) instance.classValue();

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		int noOfVals = m_ParamsPerAtt[u];
		int targetV = (int) instance.value(u);

		// find the appropriate leaf
		int depth = 0;
		while (att != -1) { // we want to consider kdb k=k
			for (int y = 0; y < nc; y++) {
				if (y != x_C)
					posteriorDist[depth][y] *= SUtils.MEsti(pt.getXYCount(targetV, y), classCounts[y], noOfVals);
				else
					posteriorDist[depth][y] *= SUtils.MEsti(pt.getXYCount(targetV, y) - 1, classCounts[y]-1, noOfVals);
			}

			int v = (int) instance.value(att);

			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;

			// check that the next node has enough examples for this value;
			int cnt = 0;
			for (int y = 0; y < nc && cnt < 2; y++) {
				cnt += next.getXYCount(targetV, y);
			}

			// In loocv, we consider minCount=1(+1), since we have to leave out i.
			if (cnt < 2){ 
				depth++;
				break;
			}

			pt = next;
			att = pt.att;
			depth++;
		} 

		for (int y = 0; y < nc; y++) {
			double mEst;
			if (y != x_C)
				mEst = SUtils.MEsti(pt.getXYCount(targetV, y), classCounts[y], noOfVals);
			else
				mEst = SUtils.MEsti(pt.getXYCount(targetV, y)-1, classCounts[y]-1, noOfVals);

			for (int k = depth; k <= m_KDB; k++){
				posteriorDist[k][y] *= mEst;
			}
		}

	}	

	public wdBayesNode getBayesNode(Instance instance, int i, int u, int[] m_Parents) {	

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;
			pt = next;
			att = pt.att;
		}

		return pt;		
	}

	public wdBayesNode getBayesNode(Instance instance, int i) {	

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null) 
				break;
			pt = next;
			att = pt.att;
		}

		return pt;		
	}

	public void cleanUp(int m_BestattIt, int m_BestK_) {

		for (int i = m_BestattIt; i < n; i++) {
			wdBayesNode_[i] = null;
		}

		for (int i = 0; i < m_BestattIt; i++) {
			if (parents[i] != null) {
				if (parents[i].length > m_BestK_) {
					int level = -1;
					deleteExtraNodes(wdBayesNode_[i], m_BestK_, level);
				}	
			}
		}
	}

	public void deleteExtraNodes(wdBayesNode pt, int k, int level) {
		
		level = level + 1;
		
		int att = pt.att;

		while (att != -1) {

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				
				if (level == k) {
					pt.children[c] = null;
					pt.att = -1;
					next = null;
				}
				
				if (next != null) 
					deleteExtraNodes(next, k, level);

				att = -1;
			}
		}

	}
	
	/* 
	 * -----------------------------------------------------------------------------------------
	 * Allocate Parameters
	 * -----------------------------------------------------------------------------------------
	 */	

	public void allocate() {
		// count active nodes in Trie
		np = nc;
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			activeNumNodes[u] = countActiveNumNodes(u, order[u], parents[u], pt);
		}		
		System.out.println("Allocating dParameters of size: " + np);
		parameters = new double[np];				
	}

	public int countActiveNumNodes(int i, int u, int[] lparents, wdBayesNode pt) {
		int numNodes = 0;		
		int att = pt.att;

		if (att == -1) {
			pt.index = np;
			np += m_ParamsPerAtt[u] * nc;			
			return 1;			
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					numNodes += countActiveNumNodes(i, u, lparents, next);
				att = -1;
			}			
		}

		return numNodes;
	}
	
	/* 
	 * -----------------------------------------------------------------------------------------
	 * xyParameters to Parameters
	 * -----------------------------------------------------------------------------------------
	 */	
	
	public void reset() {		
		// convert a trie into an array
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			trieToArray(u, order[u], parents[u], pt);
		}		
	}

	private int trieToArray(int i, int u, int[] parents, wdBayesNode pt) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					//System.out.println(index + (c * paramsPerAtt[u] + j));
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = pt.getXYParameter(j, c);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					trieToArray(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;		
	}
	
	// ----------------------------------------------------------------------------------
	// Parameters to xyParameters
	// ----------------------------------------------------------------------------------

	public void copyParameters(double[] params) {
		for (int i = 0; i < params.length; i++) {
			parameters[i] = params[i];
		}		

		// convert an array into a trie
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			arrayToTrie(u, order[u], parents[u], pt);			
		}		
	}

	private int arrayToTrie(int i, int u, int[] parents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = parameters[index + (c * m_ParamsPerAtt[u] + j)];
					pt.setXYParameter(j, c, val);
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					arrayToTrie(i, u, parents, next);
				att = -1;
			}			
		}

		return 0;	
	}
	
	// ----------------------------------------------------------------------------------
	// update xyParameters based on eps * gradient 
	// ----------------------------------------------------------------------------------
	
	public void updateParameters(double alpha, double[] grad) {
		for (int c = 0; c < nc; c++) {
			parameters[c] = parameters[c] - (alpha * grad[c]);
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			updateParameters(u, order[u], parents[u], pt, alpha, grad);
		}
	}
	
	private int updateParameters(int i, int u, int[] parents, wdBayesNode pt, double alpha, double[] grad) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					int location = index + (c * m_ParamsPerAtt[u] + j);
					
					double newval = pt.getXYParameter(j, c) - (alpha * grad[location]);
					
					pt.setXYParameter(j, c, newval);	
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = newval;
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					updateParameters(i, u, parents, next, alpha, grad);
				att = -1;
			}			
		}

		return 0;		
	}
	
	public void updateParameters(double[] stepSize, double[] grad) {
		for (int c = 0; c < nc; c++) {
			parameters[c] = parameters[c] - (stepSize[c] * grad[c]);
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			updateParameters(u, order[u], parents[u], pt, stepSize, grad);
		}
	}
	
	private int updateParameters(int i, int u, int[] parents, wdBayesNode pt, double[] stepSize, double[] grad) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					int location = index + (c * m_ParamsPerAtt[u] + j);
					
					double newval = pt.getXYParameter(j, c) - (stepSize[location] * grad[location]);
					
					pt.setXYParameter(j, c, newval);	
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = newval;
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					updateParameters(i, u, parents, next, stepSize, grad);
				att = -1;
			}			
		}

		return 0;		
	}
	
	// ----------------------------------------------------------------------------------
	// initialize xyParameters with val 
	// ----------------------------------------------------------------------------------

	public void initializeParametersWithVal(double initVal) {
		for (int c = 0; c < nc; c++) {
			parameters[c] = initVal;
		}
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			initializeParametersWithVal(u, order[u], parents[u], pt, initVal);
		}
	}

	private int initializeParametersWithVal(int i, int u, int[] parents, wdBayesNode pt, double initVal) {		
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					pt.setXYParameter(j, c, initVal);	
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = initVal;
				}				
			}			
			return 0;
		}			

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					initializeParametersWithVal(i, u, parents, next, initVal);
				att = -1;
			}			
		}

		return 0;		
	}

	public double getNLL_MAP(Instances instances) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc); 
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = classProbabilities[c];
				//myProbs[c] = xyDist.pp(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYProbability((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			//nll += (- myProbs[x_C]);
		}

		return nll;
	}
	
	public double[] getParameters() {
		return parameters;
	}

	public int getNp() {
		return np;
	}

	public int setNAttributes(int newn) {
		return n = newn;
	}
	
	public int getNAttributes() {
		return n;
	}

	public double[] getClassCounts() {
		return classCounts;
	}

	public double[] getClassProbabilities() {
		return classProbabilities;
	}


}