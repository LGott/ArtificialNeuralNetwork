package artificialNeuralNetwork;

/**This class is released under the limited GNU public
* license (LGPL).
*
* Author: Jeff Heaton
* Modified by Leba Gottesman 
* **/

import java.text.*;

public class TestANN {
	public static void main(String args[]) {
		// Input matrix for 7-segment-display digits 0-9
		double input[][] = { { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 }, { 0.0, 1.0, 1, 0, 0.0, 0.0, 0.0, 0.0 },
				{ 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 }, { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0 },
				{ 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 }, { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0 },
				{ 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
				{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0 } };

		// Corresponding output
		double idealOutput[][] = { { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
				{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
				{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 } };

		System.out.println("Train:");

		int inputAmnt = 7;
		int hiddenAmnt = 4;
		int outputAmnt = 10;
		double learningRate = 1.0;
		double momentum = 0.9;

		Network network = new Network(inputAmnt, hiddenAmnt, outputAmnt, learningRate, momentum);

		NumberFormat percentFormat = NumberFormat.getPercentInstance();
		percentFormat.setMinimumFractionDigits(4);

		int counter = 0;
		for (int i = 0; i < 10000; i++) {
			for (int j = 0; j < input.length; j++) {
				network.computeOutputs(input[j]);
				network.calcError(idealOutput[j]);
				network.learn();
			}

			System.out.println("Trial #" + counter + ", Error:" + percentFormat.format(network.getError(input.length)));
			counter++;
		}

		System.out.println("\nTest Accuracy:");

		for (int i = 0; i < input.length; i++) {
			System.out.println();
			System.out.print(i + " = [ ");
			for (int j = 0; j < input[0].length; j++) {
				System.out.print(input[i][j] + ",");
			}
			System.out.print(" ]");
			double out[] = network.computeOutputs(input[i]);

			System.out.print("=> [ ");
			for (int p = 0; p < out.length; p++) {
				System.out.printf(",  " + "%.2f", out[p]);

			}
			System.out.print(" ] ");
		}
		System.out.println("\n\nInputs: " + inputAmnt + "\nHidden: " + hiddenAmnt + "\nOutputs: " + outputAmnt
				+ "\nLearning Rate: " + learningRate + "\nMomentum: " + momentum);
	}
}