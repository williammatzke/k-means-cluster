
/**
 * A k-means clustering algorithm implementation.
 * 
 */

import java.util.ArrayList;

public class KMeans {
	public KMeansResult cluster(double[][] centroids, double[][] instances, double threshold) {

		ArrayList<Double> distortions = new ArrayList<Double>();
		int[] clusterAssignment = new int[instances.length];

		while (aboveThreshold(distortions, threshold)) {
			// allocate clusters to nearest centroid
			for (int i = 0; i < instances.length; i++) {
				clusterAssignment[i] = findNearestCentroid(centroids, instances[i]);
			}

			// check for and deal with orphans
			while (findOrphan(clusterAssignment, centroids.length) != -1) {
				int orphan = findOrphan(clusterAssignment, centroids.length);
				centroids[orphan] = findWorstPoint(centroids, instances, clusterAssignment);
				// reallocate clusters to nearest centroid
				for (int i = 0; i < instances.length; i++) {
					clusterAssignment[i] = findNearestCentroid(centroids, instances[i]);
				}
			}

			centroids = calculateCentroids(centroids, instances, clusterAssignment);

			distortions.add(calculateDistortion(centroids, instances, clusterAssignment));
		}

		// format and return result
		KMeansResult result = new KMeansResult();
		result.centroids = centroids;
		result.clusterAssignment = clusterAssignment;
		result.distortionIterations = new double[distortions.size()];
		for (int i = 0; i < distortions.size(); i++) {
			result.distortionIterations[i] = distortions.get(i);
		}
		return result;
	}

	/**
	 * Find the worst instance/point for dealing with orphans
	 * 
	 * @param centroids
	 * @param instances
	 * @param clusterAssignment
	 * @return d dimension array containing the feature space of the farthest
	 *         point
	 */
	private double[] findWorstPoint(double[][] centroids, double[][] instances, int[] clusterAssignment) {
		double worstDistance = 0;
		int worstIndex = -1;
		for (int i = 0; i < instances.length; i++) {
			double currentDistance = euclideanSquaredDistance(centroids[clusterAssignment[i]], instances[i]);
			if (currentDistance > worstDistance) {
				worstDistance = currentDistance;
				worstIndex = i;
			}
		}
		return instances[worstIndex];
	}

	/**
	 * @param centroids
	 * @param instances
	 * @return The total distortion between all centroids and all instances
	 */
	private double calculateDistortion(double[][] centroids, double[][] instances, int[] clusterAssignments) {
		// go through the first dimension and calcualte the euclidean distance
		// row by row. Keep a running total
		double distortionTotal = 0.0;
		for (int i = 0; i < instances.length; i++) {
			distortionTotal += euclideanSquaredDistance(centroids[clusterAssignments[i]], instances[i]);
		}
		return distortionTotal;
	}

	/**
	 * For relocating centroids based on their center of mass
	 * 
	 * @param centroids
	 * @param instances
	 * @param clusterAssignments
	 * @return a new double[][] with for improved centroids
	 */
	private double[][] calculateCentroids(double[][] centroids, double[][] instances, int[] clusterAssignments) {
		// zero all the centroids
		for (int i = 0; i < centroids.length; i++) {
			for (int j = 0; j < centroids[0].length; j++) {
				centroids[i][j] = 0.0;
			}
		}
		// make a running total of location in appropriate dimension
		for (int a = 0; a < instances.length; a++) {
			for (int b = 0; b < instances[0].length; b++) {
				centroids[clusterAssignments[a]][b] += instances[a][b];
			}
		}
		// for the number of centroids, make a running total of how many times
		// that cluster is assigned. once that is doen for one centroid, divide.
		for (int i = 0; i < centroids.length; i++) {
			int useCount = 0;
			for (int j = 0; j < instances.length; j++) {
				if (clusterAssignments[j] == i) {
					useCount++;
				}
			}
			// go through every dimension and divide
			for (int k = 0; k < centroids[0].length; k++) {
				centroids[i][k] = centroids[i][k] / useCount;
			}
		}
		return centroids;
	}

	/**
	 * Will return the location of nearest centroid
	 * 
	 * @param centroids
	 * @param instance
	 *            one instance to look at
	 * @return
	 */
	private int findNearestCentroid(double[][] centroids, double[] instance) {
		int centroidIndex = -1;
		double bestDistance = -1;
		double currDistance = -1;
		for (int i = 0; i < centroids.length; i++) {
			currDistance = euclideanSquaredDistance(centroids[i], instance);
			if (currDistance < bestDistance || bestDistance == -1) {
				bestDistance = currDistance;
				centroidIndex = i;
			}
		}
		return centroidIndex;
	}

	/**
	 * Computes euclidean distance between two same-dimensioned clusters
	 * 
	 * @param centroid
	 * @param instance
	 * @return double distance
	 */
	private double euclideanSquaredDistance(double[] centroid, double[] instance) {
		if (centroid.length != instance.length) {
			System.out.println("Dimension number of centroids and clusters must be the same");
			return 0.0;
		} else {
			double distance = 0.0;
			for (int i = 0; i < centroid.length; i++) {
				distance += (centroid[i] - instance[i]) * (centroid[i] - instance[i]);
			}
			return distance;
		}
	}

	/**
	 * 
	 * @param clusterAssignment
	 *            array where eg clusterAssigmnent[0] returns 1, meaning that
	 *            the first cluster is assigned to centroid 1.
	 * @param centroidCount
	 *            Number of centroids being used (k). Need to find every
	 *            centroid used, otherwise it's an orphan
	 * @return -1 if no orphans, else returns the number of the first centroid
	 *         found
	 */
	private int findOrphan(int[] clusterAssignment, int centroidCount) {
		boolean[] centroidUsed = new boolean[centroidCount];
		for (int i = 0; i < centroidUsed.length; i++) {
			centroidUsed[i] = false;
		}
		for (int i = 0; i < clusterAssignment.length; i++) {
			centroidUsed[clusterAssignment[i]] = true;
		}
		for (int i = 0; i < centroidUsed.length; i++) {
			if (centroidUsed[i] == false) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * determines whether the change in distortion has decreased below the
	 * threshold yet
	 * 
	 * @param distortions
	 * @param threshold
	 * @return
	 */
	private boolean aboveThreshold(ArrayList<Double> distortions, double threshold) {
		if (distortions.size() < 2) {
			return true;
		} else {
			double change = Math.abs((distortions.get(distortions.size() - 1) - distortions.get(distortions.size() - 2))
					/ distortions.get(distortions.size() - 2));
			return (change > threshold);
		}
	}
}
