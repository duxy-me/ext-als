package main;

import java.io.IOException;

import data_structure.DenseMatrix;
import utils.Printer;
import algorithms.MF_extALS;
import algorithms.MF_eALS;
import algorithms.MF_eALS_bigW;
import algorithms.ItemPopularity;

public class main_MF extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "yelp";
		String method = "extALS";
		double w0 = 10;
		boolean showProgress = false;
		boolean showLoss = true;
		int factors = 64;
		int maxIter = 500;
		double reg = 0.01;
		double alpha = 0.75;
		int relate = 1;

		// sample: java -jar extALS.jar yelp extALS w0 false true 64 500 0.01 0 relates
		if (argv.length > 0) {
			dataset_name = argv[0];
			method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8)
				alpha = Double.parseDouble(argv[8]);
			if (argv.length > 9)
				relate = Integer.parseInt(argv[9]);
		}

		ReadRatings_HoldOneOut("data/" + dataset_name + ".rating");

		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f, relate=%d\n",
				method, showProgress, factors, maxIter, reg, w0, alpha, relate);
		System.out.println("====================================================");

		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");

		double init_mean = 0;
		double init_stdev = 0.01;

		if (method.equalsIgnoreCase("eals")) {
			System.out.println("training size: " + trainMatrix.size());
			MF_eALS eals = new MF_eALS(trainMatrix, testRatings, topK, threadNum, factors, maxIter, w0, alpha, reg,
					init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(eals, "MF_eALS");
		}
		if (method.equalsIgnoreCase("extals")) {
			System.out.println("training size: " + trainMatrix.size());
			MF_extALS eals = new MF_extALS(trainMatrix, testRatings, topK, threadNum, factors, relate, maxIter, w0,
					alpha, reg, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(eals, "MF_extALS");
		}

		if (method.equalsIgnoreCase("eals_bigw")) {
			System.out.println("training size: " + trainMatrix.size());
			MF_eALS_bigW eals = new MF_eALS_bigW(trainMatrix, testRatings, topK, threadNum, factors, maxIter, w0, alpha,
					reg, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(eals, "MF_eALS_faster");
		}

	} // end main
}
