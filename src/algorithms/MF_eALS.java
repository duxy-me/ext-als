package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;
import data_structure.SparseVector;
import happy.coding.math.Randoms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.HashMap;

import utils.LatentFactorUtil;
import utils.Printer;

/**
 * element wised ALS for weighted matrix factorization (with imputation)
 * 
 * @author Xiaoyu Du
 */
public class MF_eALS extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; // number of latent factors.
	int maxIter = 500; // maximum iterations.
	double reg = 0.01; // regularization parameters
	double w0 = 1;
	double init_mean = 0; // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V
	int relates = 1; // dimensions of negative weights, i.e. Z in paper

	double[] prediction_items;
	double[] rating_items;
	double[] w_items;
	double[] prediction_users;
	double[] rating_users;
	double[] w_users;

	/** Model parameters to learn */
	public DenseMatrix U; // latent vectors for users
	public DenseMatrix V; // latent vectors for items

	/** Caches */
	DenseMatrix SU;
	DenseMatrix SV;

	boolean showProgress;
	boolean showLoss;

	// weight for each positive instance in trainMatrix
	DenseMatrix W;

	// weight for negative instances on item i, user u.
	DenseMatrix Wi, Wu;

	public MF_eALS(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, int topK, int threadNum, int factors,
			int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev,
			boolean showProgress, boolean showLoss) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showLoss = showLoss;
		this.showProgress = showProgress;

		// Set the Wi as a decay function w0 * pi ^ alpha
		double sum = 0, Z = 0;
		double[] p = new double[itemCount];
		for (int i = 0; i < itemCount; i++) {
			p[i] = trainMatrix.getColRef(i).itemCount();
			sum += p[i];
		}
		// convert p[i] to probability
		for (int i = 0; i < itemCount; i++) {
			p[i] /= sum;
			p[i] = Math.pow(p[i], alpha);
			Z += p[i];
		}
		// assign weight
		Wu = new DenseMatrix(userCount, relates);
		Wi = new DenseMatrix(itemCount, relates);
		for (int u = 0; u < userCount; u++) {
			for (int t = 0; t < relates; t++) {
				Wu.set(u, t, 1);
			}
		}
		for (int i = 0; i < itemCount; i++) {
			for (int t = 0; t < relates; t++) {
				Wi.set(i, t, w0 * p[i] / Z);
			}
		}

		prediction_items = new double[itemCount];
		rating_items = new double[itemCount];
		w_items = new double[itemCount];
		prediction_users = new double[userCount];
		rating_users = new double[userCount];
		w_users = new double[userCount];

		// By default, the weight for positive instance is uniformly 1.
		W = Wu.mult(Wi.transpose());

		// should we initialized the rated weight to 1? to be done
		for (int u = 0; u < userCount; u++)
			for (int i : trainMatrix.getRowRef(u).indexList())
				W.set(u, i, 1);

		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
	}

	// remove
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
	}

	public void buildModel() {
		System.out.printf("user: %d, item: %d\n", userCount, itemCount);
		// System.out.println("Run for FastALS. ");
		double loss_pre = Double.MAX_VALUE;
		for (int iter = 0; iter < maxIter; iter++) {
			Long start = System.currentTimeMillis();

			// Update user latent vectors
			for (int u = 0; u < userCount; u++) {
				update_user(u);
			}

			// Update item latent vectors
			for (int i = 0; i < itemCount; i++) {
				update_item(i);
			}

			// Show progress
			if (showProgress)
				showProgress(iter, start, testRatings);
			// Show loss
			if (showLoss)
				loss_pre = showLoss(iter, start, loss_pre);

		} // end for iter
	}

	// Run model for one iteration
	public void runOneIteration() {
		// Update user latent vectors
		for (int u = 0; u < userCount; u++) {
			update_user(u);
		}

		// Update item latent vectors
		for (int i = 0; i < itemCount; i++) {
			update_item(i);
		}
	}

	protected void update_user(int u) {
		for (int i = 0; i < itemCount; i++) {
			prediction_items[i] = predict(u, i);
			rating_items[i] = trainMatrix.getValue(u, i);
			w_items[i] = W.get(u, i);
		}
		for (int f = 0; f < factors; f++) {
			double numer = 0.;
			double denom = 0.;
			for (int i = 0; i < itemCount; i++) {
				prediction_items[i] -= U.get(u, f) * V.get(i, f);

				numer += (rating_items[i] - prediction_items[i]) * w_items[i] * V.get(i, f);
				denom += w_items[i] * V.get(i, f) * V.get(i, f);
			}
			denom += reg;
			U.set(u, f, numer / denom);

			for (int i = 0; i < itemCount; i++)
				prediction_items[i] += U.get(u, f) * V.get(i, f);
		} // end for f
	}

	protected void update_item(int i) {
		for (int u = 0; u < userCount; u++) {
			prediction_users[u] = predict(u, i);
			rating_users[u] = trainMatrix.getValue(u, i);
			w_users[u] = W.get(u, i);
		}

		for (int f = 0; f < factors; f++) {
			// O(K) complexity for the w0 part
			double numer = 0, denom = 0;
			for (int u = 0; u < userCount; u++) {
				prediction_users[u] -= U.get(u, f) * V.get(i, f);

				numer += (rating_users[u] - prediction_users[u]) * w_users[u] * U.get(u, f);
				denom += w_users[u] * U.get(u, f) * U.get(u, f);
			}
			denom += reg;
			V.set(i, f, numer / denom);

			for (int u = 0; u < userCount; u++)
				prediction_users[u] += U.get(u, f) * V.get(i, f);
		} // end for f
	}

	public double showLoss(int iter, long start, double loss_pre) {
		long start1 = System.currentTimeMillis();
		double loss_cur = loss();
		String symbol = loss_pre >= loss_cur ? "-" : "+";
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s]\n", iter, Printer.printTime(start1 - start), symbol,
				loss_cur, Printer.printTime(System.currentTimeMillis() - start1));
		return loss_cur;
	}

	public double loss() {
		double L = reg * (U.squaredSum() + V.squaredSum());

		for (int u = 0; u < userCount; u++) {
			for (int i = 0; i < itemCount; i++) {
				L += W.get(u, i) * Math.pow(predict(u, i) - trainMatrix.getValue(u, i), 2);
			}
		}
		System.out.println("total = " + L);
		return L;
	}

	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}

	@Override
	public void updateModel(int u, int i) {
		// not implemented
	}

}
