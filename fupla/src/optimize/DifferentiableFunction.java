package optimize;

/**
 * Class responsible for returning the function values at given point
 * @author Mateusz Kobos
 */
public interface DifferentiableFunction {
	/**
	 * @param point point of the function evaluation
	 * @return values in given point if the algorithm should continue 
	 * computations or null if the algorithm should stop
	 */
	FunctionValues getValues(double[] point);
	
	public void finish();

	public int get_nr_variable();

	public double fun();

	public double fun(double[] point);

	public void grad(double[] grad);

	public void Hv(double[] s, double[] Hs);

	void initializeParameters(double[] w_new);
}
