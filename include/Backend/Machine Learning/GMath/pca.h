// Copyright 2024 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <vector>
#include <algorithm>
#include <cmath>
#include <cmath>
#include <fstream>
#include <iostream>

// Helper function to compute the mean of a vector of numbers
double compute_mean(const std::vector<double>& data);

// Helper function to compute the dot product of two vectors
double dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2);

// Helper function to perform matrix-vector multiplication
std::vector<double> matrix_vector_multiply(const std::vector<std::vector<double> >& matrix, const std::vector<double>& vec);

// Custom comparison function for sorting in descending order
bool compare_pairs(const std::pair<double, std::vector<double> >& pair1, const std::pair<double, std::vector<double> >& pair2);

// Multiply two matrices: C = A * B
std::vector<std::vector<double> > matrixMultiply(const std::vector<std::vector<double> >& A,
	const std::vector<std::vector<double> >& B);

// Gram-Schmidt orthogonalization
void gramSchmidt(std::vector<std::vector<double> >& matrix);

// Main function to compute PCA
void compute_pca(const std::vector<std::vector<double> >& data, std::vector<std::vector<double> >& transformed_data, std::vector<std::vector<double> >& sorted_eig_vecs);

void calculate_arrow_head(double x1, double y1, double x2, double y2);


// CALL THIS
// Top level function
void pca(const std::vector<std::vector<double> >& data, std::vector<std::vector<double> >& transformed_data, std::vector<std::vector<double> >& sorted_eig_vecs);
