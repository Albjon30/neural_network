package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func main() {
	rand.NewSource(time.Now().UnixNano())

	// Training data for OR logic gate
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := []float64{0, 1, 1, 1}

	// Initialize weights and bias
	weights := []float64{rand.Float64(), rand.Float64()}
	bias := rand.Float64()
	learningRate := 0.1

	// Train the network
	for epoch := 0; epoch < 10000; epoch++ {
		for i := range inputs {
			// Forward pass
			sum := inputs[i][0]*weights[0] + inputs[i][1]*weights[1] + bias
			output := sigmoid(sum)

			// Error
			error := targets[i] - output

			// Backpropagation
			delta := error * sigmoidDerivative(output)
			weights[0] += inputs[i][0] * delta * learningRate
			weights[1] += inputs[i][1] * delta * learningRate
			bias += delta * learningRate
		}
	}

	// Test
	fmt.Println("Trained Neural Network:")
	for i := range inputs {
		sum := inputs[i][0]*weights[0] + inputs[i][1]*weights[1] + bias
		output := sigmoid(sum)
		fmt.Printf("Input: %v => Output: %.4f\n", inputs[i], output)
	}
}
