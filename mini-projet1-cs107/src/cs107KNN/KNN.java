package cs107KNN;

import java.util.Arrays;

public class KNN {
	public static void main(String[] args) {

		test(10,1);
		
		/*byte b1 = 40; // 00101000
		byte b2 = 20; // 00010100
		byte b3 = 10; // 00001010							//test de début
		byte b4 = 5; // 10000101

		// [00101000 | 00010100 | 00001010 | 00000101] = 672401925
		int result = extractInt(b1, b2, b3, b4);
		System.out.println(result);

		String bits = "10000001";
		System.out.println("La séquence de bits " + bits + "\n\tinterprétée comme byte non signé donne "
				+ Helpers.interpretUnsigned(bits) + "\n\tinterpretée comme byte signé donne "
				+ Helpers.interpretSigned(bits));
		System.out.println(Helpers.byteToBinaryString (b1));*/
	}
	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param bXToBY The byte containing the bits to store between positions X and Y
	 * 
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {

		return ((b31ToB24)&0xFF) << 24 | (b23ToB16&0xFF) << 16 | (b15ToB8&0xFF) << 8 | (b7ToB0&0xFF);

	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {
		if(extractInt(data[0], data[1], data[2], data[3]) != 2051) {
			return null;
		} else {
		int nombreImages = extractInt(data[4], data[5], data[6], data[7]);
		int hauteurImage = extractInt(data[8], data[9], data[10], data[11]);
		int largeurImage = extractInt(data[12], data[13], data[14],data[15]);
		byte[][][] tensor = new byte[nombreImages][hauteurImage][largeurImage];

		for (int i = 16; i < data.length; i++) {
			byte pixel=data[i];
			byte pixelValue = (byte) ((pixel & 0xFF) - 128) ;
			data[i]=pixelValue;
		}

		for (int i = 0; i < nombreImages; i++) {
			for (int j = 0; j < hauteurImage; j++) {
				for (int j2 = 0; j2 < largeurImage; j2++) {
					tensor[i][j][j2]=data[16+j2+i*largeurImage*hauteurImage+j*largeurImage];
				}
			}
		}
		return tensor;
		}
	}

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {
		if(extractInt(data[0], data[1], data[2], data[3]) != 2049) {
			return null;
		} else {
		int nombreLabels = extractInt(data[4], data[5], data[6], data[7]);
		byte[] tensor= new byte[nombreLabels];
		for (int i = 0; i < data.length; i++) {
			if (i<data.length-8) {
				tensor[i]= data[i+8];
			}
		}
		return tensor;
		}
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {

		float distance = 0;
		int hauteurImage = a.length;
		int largeurImage = a[0].length;
		for (int i = 0; i < hauteurImage-1; i++) {
			for (int j = 0; j < largeurImage-1; j++) {
				distance+=(a[i][j]-b[i][j])*(a[i][j]-b[i][j]);
			}
		}
		distance*=distance;
		return distance;
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {

		double si1 = 0;
		float moyenneA=0;
		float moyenneB =0;					//défini valeur
		double dessus =0;
		double dessousA =0;
		double dessousB =0;
		int hauteurImage = a.length;
		int largeurImage = a[0].length;
		for (int i = 0; i < hauteurImage-1; i++) {  					//calcul moyenne A
			for (int j = 0; j < largeurImage-1; j++) {
				moyenneA+=a[i][j];
			}
		}
		moyenneA*=1/(hauteurImage*largeurImage);
		for (int i = 0; i < hauteurImage-1; i++) {						//calcul moyenne B
			for (int j = 0; j < largeurImage-1; j++) {
				moyenneB+=b[i][j];
			}
		}
		moyenneB*=1/(hauteurImage*largeurImage);

		for (int i = 0; i < hauteurImage-1; i++) {						//calcul le dessus de la fraction
			for (int j = 0; j < largeurImage-1; j++) {
				dessus+=(a[i][j] -moyenneA)*(b[i][j]-moyenneB);
			}
		}
		for (int i = 0; i < hauteurImage-1; i++) {						//calcul dessous A fraction
			for (int j = 0; j < largeurImage-1; j++) {
				dessousA+=(a[i][j]-moyenneA)*(a[i][j]-moyenneA);
			}
		}

		for (int i = 0; i < hauteurImage-1; i++) {						//calcul dessous B fration 
			for (int j = 0; j < largeurImage-1; j++) {
				dessousB+=(b[i][j]-moyenneB)*(b[i][j]-moyenneB);
			}
		}


		double dessous = Math.sqrt(dessousA*dessousB);
		si1=1-(dessus/dessous);

		float f = (float) si1;
		return f;
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 * 
	 * @param values the values whose indices have to be sorted in non decreasing
	 *               order
	 * 
	 * @return the array of sorted indices
	 * 
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */	
	public static int[] quicksortIndices(float[] values) {

		int[] indices = new int [values.length];   					//créer tableau indices
		for (int i = 0; i < indices.length; i++) {
			indices[i]=i;
		}
		quicksortIndices(values,indices, 0,values.length-1);
		return indices;
	}
	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * 
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
		int l = low;   
		int h = high ;
		float pivot =  values[low];   
		do {
			if (values[l] < pivot) {
				l++;
			}
			else if (values[h] > pivot) { 
				h--;
			}
			else {
				swap(l,h, values,indices);
				l++;
				h--;												//}Algorithm donné
			}
		}while (l <= h) ;
		if (low < h) { 
			quicksortIndices(values, indices, low, h);
		}
		if (values[high] > values[l]) {
			quicksortIndices(values, indices, l, high);
		}

	}


	/**

	/**
	 * @brief Swaps the elements of the given arrays at the provided positions
	 * 
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {
		float temp=values[i];
		values[i]=values[j];
		values[j]=temp;
		int temp2=indices[i];			//swap classic mais à double pour changer valeur et indice
		indices[i]=indices[j];
		indices[j]=temp2;
	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {
		int i = 0;
		int n = 1;
		do {
			if(array[i] < array[n]) {
				i = n;
				n++;
			} else {
				n+= 1;
			}

		}while (n < array.length);
		return i;
	}//compare les valeurs du tableau et retourne l'indice de la valeur du tableau la plus grande.

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
		int[] essai = new int[10]; //tableau pour stocker les votes
		for(int i = 0; i < k; i++) {
			essai[labels[sortedIndices[i]]] = essai[labels[sortedIndices[i]]] + 1;
		}
		int resultat = indexOfMax(essai); //cherche la plus grande valeur (+ de votes)
		return (byte) resultat;
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {
		float distances[] = new float[trainImages.length];
		for(int i = 0; i < trainImages.length; i++) {
			distances[i] = invertedSimilarity(image, trainImages[i]);
		}
		int Indices[] = quicksortIndices(distances);
		byte resultat = electLabel(Indices, trainLabels, k);

		return resultat;
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
		// TODO: Implémenter
		double a = 0;
		double n = trueLabels.length;
		for(int i = 0; i < n; i++) {
			if(predictedLabels[i] == trueLabels[i]) {
				a += (1/n);
			}

		}
		return a;
	}
	
	public static void test(int TESTS, int K) {
		
		byte[][][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/5000-per-digit_images_train")) ;
		byte[] trainLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/5000-per-digit_labels_train")) ;
		byte[][][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test")) ;
		byte[] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test")) ;
		byte[] predictions = new byte[TESTS] ;
		long start = System.currentTimeMillis () ; 	//calcul du temps 
		int[] rate = new int [TESTS];
		int d=0;
		for (int i = 0 ; i < TESTS ; i++) {
			predictions[i] = knnClassify(testImages[i], trainImages , trainLabels , K) ;
			/*System.out.println("Test n°" + i);
			if (predictions[i]==testLabels[i]) {
				System.out.println(" réussi");
			}
			else {
				System.out.println(" raté");
				rate[d]=i;
			}
			d=d+1;*/
		}
		long end = System.currentTimeMillis () ;
		double time = (end - start) / 1000d ;
		int e=0;
		for (int i = 0; i < rate.length; i++) {
			if (rate[i]!=0) {
			System.out.println("Le test n°" + rate[i]+ " a echoué. Nous attendions un "+ 
			testLabels[i] +" alors que l'ordinateur a prédit un " + predictions[i] + ".");
			e+=1;
			}
		}
		if(e==0) {
			System.out.println("---Aucun test n'a échoué !---");
		}
		System.out.println("Accuracy = " + accuracy(predictions , Arrays.copyOfRange(testLabels , 0, TESTS))*100 + " %") ;
		System.out.println("Time = " + time + " seconds") ;
		System.out.println("Time per test image = " + (time / TESTS)) ;

		Helpers.show("Test", testImages , predictions , testLabels , (int)(Math.sqrt(TESTS)), (int)(Math.sqrt(TESTS))) ;
	}
}
