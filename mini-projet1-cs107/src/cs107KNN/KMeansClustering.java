package cs107KNN;

import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.ArrayList;

public class KMeansClustering {
	public static void main(String[] args) {
		int K = 7;
		int maxIters = 20;
		byte[] test = new byte[8];
		encodeInt(2051,test,0);
		encodeInt(2049,test,4);

		for (int i = 0; i < test.length; i++) {
			System.out.print(test[i]);
		}
		



		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/1000-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/1000-per-digit_labels_train"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		byte[] reducedLabels = new byte[reducedImages.length];
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}

		Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));
		 KNN.test(100, 7, true);
		 
	}

	/**
	 * @brief Encodes a tensor of images into an array of data ready to be written on a file
	 * 
	 * @param images the tensor of image to encode
	 * 
	 * @return the array of byte ready to be written to an IDX file
	 */
	public static byte[] encodeIDXimages(byte[][][] images) {
		int nbrImage = images.length;
		int hauteurImage = images[0].length;
		int largeurImage = images[0][0].length;
		byte[] idx = new byte[images.length * hauteurImage * largeurImage + 16];
		encodeInt(2051, idx, 0);
		encodeInt(images.length, idx, 4);
		encodeInt(images[0].length, idx, 8);
		encodeInt(images[0][0].length, idx, 12);
		
		
		for(int i = 0; i < images.length; i++) {
			for (int j = 0; j < images[i].length; j++) {
				for (int j2 = 0; j2 < images[i][j].length; j2++) {
					idx[16+j2+i*largeurImage*hauteurImage+j*largeurImage]=images[i][j][j2];
				}
			}
		}
		
		//remets les bits en non signés
		for (int i = 16; i < idx.length; i++) {
			byte pixel=idx[i];
			byte pixelValue = (byte) ((pixel & 0xFF) + 128) ;
			idx[i]=pixelValue;
		}
		return idx;
	}

	/**
	 * @brief Prepares the array of labels to be written on a binary file
	 * 
	 * @param labels the array of labels to encode
	 * 
	 * @return the array of bytes ready to be written to an IDX file
	 */
	public static byte[] encodeIDXlabels(byte[] labels) {
		byte[] idx = new byte[labels.length + 8];
		encodeInt(2049, idx, 0);
		encodeInt(labels.length, idx, 4);
		for(int i = 0; i < labels.length; i++) {
			idx[i+8] = labels[i];
		}
		return idx;
	}

	/**
	 * @brief Decomposes an integer into 4 bytes stored consecutively in the destination
	 * array starting at position offset
	 * 
	 * @param n the integer number to encode
	 * @param destination the array where to write the encoded int
	 * @param offset the position where to store the most significant byte of the integer,
	 * the others will follow at offset + 1, offset + 2, offset + 3
	 */

	public static byte[] encodeInt(int n, byte[] destination, int offset) {

		/*destination[offset] = (byte) (n&0xFF);
		destination[offset + 1] = (byte) ((n&0xFF) >> 8);
		destination[offset + 2] = (byte) ((n&0xFF) >> 16);
		destination[offset + 3] = (byte) ((n&0xFF) >> 24); */ //inverse de extractInt
		//prend les 8 premiers bites d'un int, stocke dans le tableau, ensuite les huit suivants, etc...
		destination[offset] = (byte) ((n) >>24);
		destination[offset + 1] = (byte) ((n) >> 16);
		destination[offset + 2] = (byte) ((n) >> 8);
		destination[offset + 3] = (byte) ((n) >> 0);
		return destination;
	}

	/**
	 * @brief Runs the KMeans algorithm on the provided tensor to return size elements.
	 * 
	 * @param tensor the tensor of images to reduce
	 * @param size the number of images in the reduced dataset
	 * @param maxIters the number of iterations of the KMeans algorithm to perform
	 * 
	 * @return the tensor containing the reduced dataset
	 */
	public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {
		int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
		byte[][][] centroids = new byte[size][][];
		initialize(tensor, assignments, centroids);

		int nIter = 0;
		while (nIter < maxIters) {
			// Step 1: Assign points to closest centroid
			recomputeAssignments(tensor, centroids, assignments);
			System.out.println("Recomputed assignments");
			// Step 2: Recompute centroids as average of points
			recomputeCentroids(tensor, centroids, assignments);
			System.out.println("Recomputed centroids");

			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

			nIter++;
		}

		return centroids;
	}

	/**
	 * @brief Assigns each image to the cluster whose centroid is the closest.
	 * It modifies.
	 * 
	 * @param tensor the tensor of images to cluster
	 * @param centroids the tensor of centroids that represent the cluster of images
	 * @param assignments the vector indicating to what cluster each image belongs to.
	 *  if j is at position i, then image i belongs to cluster j
	 */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
		float distances[] = new float[centroids.length];
		for(int i = 0; i < centroids.length; i++) {
			for(int j = 0; j < centroids.length; j++) {
				distances[j] = KNN.squaredEuclideanDistance(tensor[i], centroids[j]);
			}
			int[] indices = KNN.quicksortIndices(distances);
			assignments[i] = indices[0];
		}
	}

	/**
	 * @brief Computes the centroid of each cluster by averaging the images in the cluster
	 * 
	 * @param tensor the tensor of images to cluster
	 * @param centroids the tensor of centroids that represent the cluster of images
	 * @param assignments the vector indicating to what cluster each image belongs to.
	 *  if j is at position i, then image i belongs to cluster j
	 */
	public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
		for(int i = 0; i < centroids.length; i++) {
			ArrayList<Integer> temp = new ArrayList<Integer>();
			for(int j = 0; j < assignments.length; j++) {
				if(assignments[j] == i) {
					temp.add(j);
				}
			}

		}
	}

	/**
	 * Initializes the centroids and assignments for the algorithm.
	 * The assignments are initialized randomly and the centroids
	 * are initialized by randomly choosing images in the tensor.
	 * 
	 * @param tensor the tensor of images to cluster
	 * @param assignments the vector indicating to what cluster each image belongs to.
	 * @param centroids the tensor of centroids that represent the cluster of images
	 *  if j is at position i, then image i belongs to cluster j
	 */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
		Set<Integer> centroidIds = new HashSet<>();
		Random r = new Random("cs107-2018".hashCode());
		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));
		Integer[] cids = centroidIds.toArray(new Integer[] {});
		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];
		for (int i = 0; i < assignments.length; i++)
			assignments[i] = cids[r.nextInt(cids.length)];
	}
}
