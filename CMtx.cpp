#include "CMtx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "immintrin.h"
#include <thread>


namespace MyAlgebra {
	const float CMtx::ALG_PRECISION = 0.000001f;

	const std::string CMtx::AUTHOR = "Mateusz_Grzesiuk";

	const __m256 CMtx::ZERO = _mm256_setzero_ps();

	const int CMtx::SIMD_LENGTH = 8;

	const int CMtx::THREAD_NUMBER = 4;

	const int CMtx::THREAD_BORDER = 32;

	std::string CMtx::authorName()
	{
		return CMtx::AUTHOR;
	}

	CMtx::CMtx(int row_cnt, int col_cnt, float*& matrix_cnt) :
		rows(row_cnt),
		columns(col_cnt),
		matrix(matrix_cnt)
	{}

	CMtx::CMtx(int row_cnt, int col_cnt, bool rand_init) :
		rows(row_cnt),
		columns(col_cnt)
	{
		if (rows < 0)
			rows = 0;
		if (columns < 0)
			columns = 0;
		if (rand_init) {
			matrix = new float[rows * columns];
			for (int i = 0; i < rows * columns; ++i)
				matrix[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 100.0f));
		}
		else
			matrix = zeroArray(rows, columns);
	}

	CMtx::CMtx(int row_cnt, float diagonal) :
		rows(row_cnt),
		columns(row_cnt)
	{
		if (rows < 0)
			rows = 0;
		if (columns < 0)
			columns = 0;
		matrix = zeroArray(rows, columns);
		for (int i = 0; i < rows; ++i)
			matrix[i * rows + i] = diagonal;
	}

	CMtx::CMtx(const CMtx& rhs) :
		rows(rhs.rows),
		columns(rhs.columns),
		matrix(rhs.copyArray())
	{
		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&matrix[i], _mm256_loadu_ps(&rhs.matrix[i]));
		for (; i < rows * columns; ++i)
			matrix[i] = rhs.matrix[i];
	}

	float* CMtx::copyArray() const
	{
		float* m = new float[columns * rows];
		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], _mm256_loadu_ps(&matrix[i]));
		for (; i < rows * columns; ++i)
			m[i] = matrix[i];
		return m;
	}

	float* CMtx::zeroArray(int rows, int columns)
	{
		float* m = new float[rows * columns];
		int i = 0;
		for (; i < rows * columns - (rows * columns) % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], CMtx::ZERO);
		for (; i < rows * columns; ++i)
			m[i] = 0.0f;
		return m;
	}

	CMtx::CMtx(CMtx&& rhs) : columns(rhs.columns), rows(rhs.rows), matrix(rhs.matrix)
	{
		rhs.columns = NULL;
		rhs.rows = NULL;
		rhs.matrix = NULL;
	}

	CMtx::~CMtx()
	{
		if (matrix != NULL)
			delete[] matrix;
	}

	int CMtx::getColumns()
	{
		return columns;
	}

	int CMtx::getRows()
	{
		return rows;
	}

	const CMtx& CMtx::operator=(const CMtx& rhs)
	{
		if (matrix != NULL)
			delete[] matrix;
		columns = rhs.columns;
		rows = rhs.rows;
		matrix = new float[rows * columns];
		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&matrix[i], _mm256_loadu_ps(&rhs.matrix[i]));
		for (; i < rows * columns; ++i)
			matrix[i] = rhs.matrix[i];

		return *this;
	}

	const CMtx& CMtx::operator=(CMtx&& rhs)
	{
		if (&rhs == this)
			return *this;
		if (matrix != NULL)
			delete[] matrix;
		matrix = rhs.matrix;
		rhs.matrix = NULL;
		rows = rhs.rows;
		columns = rhs.columns;
		rhs.rows = NULL;
		rhs.columns = NULL;

		return *this;
	}

	const CMtx& CMtx::operator=(float diagonal) throw()
	{
		if (rows != columns)
			throw std::exception("Can't make diagonal matrix from non-square matrix");
		if (matrix != NULL)
			delete[] matrix;
		matrix = zeroArray(rows, columns);
		for (int i = 0; i < rows; ++i)
			matrix[i * rows + i] = diagonal;

		return *this;
	}

	float* CMtx::operator[](int id) throw()
	{
		if (id >= rows || id < 0)
			throw std::exception("Index out of bonds of matrix: " + id);
		float* m = &(matrix[id * columns]);
		return m;
	}

	void CMtx::multThread(int start, int end, float*& mA, float*& mB, float*& mC, int Bcolumns) const
	{
		__m256 C[12];
		__m256 B[4];
		__m256 A[3];

		float* c[12] = { new float[8], new float[8], new float[8], new float[8], new float[8], new float[8], new float[8], new float[8], new float[8], new float[8], new float[8], new float[8] };
		int j, k;
		int i = start;
		for (; i < end - (end - start) % 4; i += 4)
		{

			j = 0;
			for (; j < rows - rows % 3; j += 3)
			{
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;
				C[2] = CMtx::ZERO;
				C[3] = CMtx::ZERO;
				C[4] = CMtx::ZERO;
				C[5] = CMtx::ZERO;
				C[6] = CMtx::ZERO;
				C[7] = CMtx::ZERO;
				C[8] = CMtx::ZERO;
				C[9] = CMtx::ZERO;
				C[10] = CMtx::ZERO;
				C[11] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);
					A[2] = _mm256_loadu_ps(mA + (j + 2) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);
					B[2] = _mm256_loadu_ps(mB + (i + 2) * columns + k);
					B[3] = _mm256_loadu_ps(mB + (i + 3) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));
					C[2] = _mm256_add_ps(C[2], _mm256_mul_ps(A[0], B[2]));
					C[3] = _mm256_add_ps(C[3], _mm256_mul_ps(A[0], B[3]));
					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));
					C[5] = _mm256_add_ps(C[5], _mm256_mul_ps(A[1], B[1]));
					C[6] = _mm256_add_ps(C[6], _mm256_mul_ps(A[1], B[2]));
					C[7] = _mm256_add_ps(C[7], _mm256_mul_ps(A[1], B[3]));
					C[8] = _mm256_add_ps(C[8], _mm256_mul_ps(A[2], B[0]));
					C[9] = _mm256_add_ps(C[9], _mm256_mul_ps(A[2], B[1]));
					C[10] = _mm256_add_ps(C[10], _mm256_mul_ps(A[2], B[2]));
					C[11] = _mm256_add_ps(C[11], _mm256_mul_ps(A[2], B[3]));

				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);
				_mm256_storeu_ps(c[2], C[2]);
				_mm256_storeu_ps(c[3], C[3]);
				_mm256_storeu_ps(c[4], C[4]);
				_mm256_storeu_ps(c[5], C[5]);
				_mm256_storeu_ps(c[6], C[6]);
				_mm256_storeu_ps(c[7], C[7]);
				_mm256_storeu_ps(c[8], C[8]);
				_mm256_storeu_ps(c[9], C[9]);
				_mm256_storeu_ps(c[10], C[10]);
				_mm256_storeu_ps(c[11], C[11]);
				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];
				c[2][0] += c[2][1] + c[2][2] + c[2][3] + c[2][4] + c[2][5] + c[2][6] + c[2][7];
				c[3][0] += c[3][1] + c[3][2] + c[3][3] + c[3][4] + c[3][5] + c[3][6] + c[3][7];
				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];
				c[5][0] += c[5][1] + c[5][2] + c[5][3] + c[5][4] + c[5][5] + c[5][6] + c[5][7];
				c[6][0] += c[6][1] + c[6][2] + c[6][3] + c[6][4] + c[6][5] + c[6][6] + c[6][7];
				c[7][0] += c[7][1] + c[7][2] + c[7][3] + c[7][4] + c[7][5] + c[7][6] + c[7][7];
				c[8][0] += c[8][1] + c[8][2] + c[8][3] + c[8][4] + c[8][5] + c[8][6] + c[8][7];
				c[9][0] += c[9][1] + c[9][2] + c[9][3] + c[9][4] + c[9][5] + c[9][6] + c[9][7];
				c[10][0] += c[10][1] + c[10][2] + c[10][3] + c[10][4] + c[10][5] + c[10][6] + c[10][7];
				c[11][0] += c[11][1] + c[11][2] + c[11][3] + c[11][4] + c[11][5] + c[11][6] + c[11][7];
				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];
				mC[j * Bcolumns + i + 2] += c[2][0];
				mC[j * Bcolumns + i + 3] += c[3][0];
				mC[(j + 1) * Bcolumns + i] += c[4][0];
				mC[(j + 1) * Bcolumns + i + 1] += c[5][0];
				mC[(j + 1) * Bcolumns + i + 2] += c[6][0];
				mC[(j + 1) * Bcolumns + i + 3] += c[7][0];
				mC[(j + 2) * Bcolumns + i] += c[8][0];
				mC[(j + 2) * Bcolumns + i + 1] += c[9][0];
				mC[(j + 2) * Bcolumns + i + 2] += c[10][0];
				mC[(j + 2) * Bcolumns + i + 3] += c[11][0];

				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
					mC[j * Bcolumns + i + 2] += mA[j * columns + k] * mB[(i + 2) * columns + k];
					mC[j * Bcolumns + i + 3] += mA[j * columns + k] * mB[(i + 3) * columns + k];
					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
					mC[(j + 1) * Bcolumns + i + 1] += mA[(j + 1) * columns + k] * mB[(i + 1) * columns + k];
					mC[(j + 1) * Bcolumns + i + 2] += mA[(j + 1) * columns + k] * mB[(i + 2) * columns + k];
					mC[(j + 1) * Bcolumns + i + 3] += mA[(j + 1) * columns + k] * mB[(i + 3) * columns + k];
					mC[(j + 2) * Bcolumns + i] += mA[(j + 2) * columns + k] * mB[i * columns + k];
					mC[(j + 2) * Bcolumns + i + 1] += mA[(j + 2) * columns + k] * mB[(i + 1) * columns + k];
					mC[(j + 2) * Bcolumns + i + 2] += mA[(j + 2) * columns + k] * mB[(i + 2) * columns + k];
					mC[(j + 2) * Bcolumns + i + 3] += mA[(j + 2) * columns + k] * mB[(i + 3) * columns + k];
				}
			}
			if (rows - j == 2) {
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;
				C[2] = CMtx::ZERO;
				C[3] = CMtx::ZERO;
				C[4] = CMtx::ZERO;
				C[5] = CMtx::ZERO;
				C[6] = CMtx::ZERO;
				C[7] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);
					B[2] = _mm256_loadu_ps(mB + (i + 2) * columns + k);
					B[3] = _mm256_loadu_ps(mB + (i + 3) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));
					C[2] = _mm256_add_ps(C[2], _mm256_mul_ps(A[0], B[2]));
					C[3] = _mm256_add_ps(C[3], _mm256_mul_ps(A[0], B[3]));
					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));
					C[5] = _mm256_add_ps(C[5], _mm256_mul_ps(A[1], B[1]));
					C[6] = _mm256_add_ps(C[6], _mm256_mul_ps(A[1], B[2]));
					C[7] = _mm256_add_ps(C[7], _mm256_mul_ps(A[1], B[3]));

				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);
				_mm256_storeu_ps(c[2], C[2]);
				_mm256_storeu_ps(c[3], C[3]);
				_mm256_storeu_ps(c[4], C[4]);
				_mm256_storeu_ps(c[5], C[5]);
				_mm256_storeu_ps(c[6], C[6]);
				_mm256_storeu_ps(c[7], C[7]);
				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];
				c[2][0] += c[2][1] + c[2][2] + c[2][3] + c[2][4] + c[2][5] + c[2][6] + c[2][7];
				c[3][0] += c[3][1] + c[3][2] + c[3][3] + c[3][4] + c[3][5] + c[3][6] + c[3][7];
				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];
				c[5][0] += c[5][1] + c[5][2] + c[5][3] + c[5][4] + c[5][5] + c[5][6] + c[5][7];
				c[6][0] += c[6][1] + c[6][2] + c[6][3] + c[6][4] + c[6][5] + c[6][6] + c[6][7];
				c[7][0] += c[7][1] + c[7][2] + c[7][3] + c[7][4] + c[7][5] + c[7][6] + c[7][7];
				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];
				mC[j * Bcolumns + i + 2] += c[2][0];
				mC[j * Bcolumns + i + 3] += c[3][0];
				mC[(j + 1) * Bcolumns + i] += c[4][0];
				mC[(j + 1) * Bcolumns + i + 1] += c[5][0];
				mC[(j + 1) * Bcolumns + i + 2] += c[6][0];
				mC[(j + 1) * Bcolumns + i + 3] += c[7][0];

				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
					mC[j * Bcolumns + i + 2] += mA[j * columns + k] * mB[(i + 2) * columns + k];
					mC[j * Bcolumns + i + 3] += mA[j * columns + k] * mB[(i + 3) * columns + k];
					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
					mC[(j + 1) * Bcolumns + i + 1] += mA[(j + 1) * columns + k] * mB[(i + 1) * columns + k];
					mC[(j + 1) * Bcolumns + i + 2] += mA[(j + 1) * columns + k] * mB[(i + 2) * columns + k];
					mC[(j + 1) * Bcolumns + i + 3] += mA[(j + 1) * columns + k] * mB[(i + 3) * columns + k];
				}
			}
			else if (rows - j == 1) {
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;
				C[2] = CMtx::ZERO;
				C[3] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);
					B[2] = _mm256_loadu_ps(mB + (i + 2) * columns + k);
					B[3] = _mm256_loadu_ps(mB + (i + 3) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));
					C[2] = _mm256_add_ps(C[2], _mm256_mul_ps(A[0], B[2]));
					C[3] = _mm256_add_ps(C[3], _mm256_mul_ps(A[0], B[3]));

				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);
				_mm256_storeu_ps(c[2], C[2]);
				_mm256_storeu_ps(c[3], C[3]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];
				c[2][0] += c[2][1] + c[2][2] + c[2][3] + c[2][4] + c[2][5] + c[2][6] + c[2][7];
				c[3][0] += c[3][1] + c[3][2] + c[3][3] + c[3][4] + c[3][5] + c[3][6] + c[3][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];
				mC[j * Bcolumns + i + 2] += c[2][0];
				mC[j * Bcolumns + i + 3] += c[3][0];

				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
					mC[j * Bcolumns + i + 2] += mA[j * columns + k] * mB[(i + 2) * columns + k];
					mC[j * Bcolumns + i + 3] += mA[j * columns + k] * mB[(i + 3) * columns + k];
				}
			}
		}
		if (end - i == 3) {
			j = 0;
			for (; j < rows - rows % 3; j += 3)
			{
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;
				C[2] = CMtx::ZERO;

				C[4] = CMtx::ZERO;
				C[5] = CMtx::ZERO;
				C[6] = CMtx::ZERO;

				C[8] = CMtx::ZERO;
				C[9] = CMtx::ZERO;
				C[10] = CMtx::ZERO;


				int k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);
					A[2] = _mm256_loadu_ps(mA + (j + 2) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);
					B[2] = _mm256_loadu_ps(mB + (i + 2) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));
					C[2] = _mm256_add_ps(C[2], _mm256_mul_ps(A[0], B[2]));

					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));
					C[5] = _mm256_add_ps(C[5], _mm256_mul_ps(A[1], B[1]));
					C[6] = _mm256_add_ps(C[6], _mm256_mul_ps(A[1], B[2]));

					C[8] = _mm256_add_ps(C[8], _mm256_mul_ps(A[2], B[0]));
					C[9] = _mm256_add_ps(C[9], _mm256_mul_ps(A[2], B[1]));
					C[10] = _mm256_add_ps(C[10], _mm256_mul_ps(A[2], B[2]));


				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);
				_mm256_storeu_ps(c[2], C[2]);

				_mm256_storeu_ps(c[4], C[4]);
				_mm256_storeu_ps(c[5], C[5]);
				_mm256_storeu_ps(c[6], C[6]);

				_mm256_storeu_ps(c[8], C[8]);
				_mm256_storeu_ps(c[9], C[9]);
				_mm256_storeu_ps(c[10], C[10]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];
				c[2][0] += c[2][1] + c[2][2] + c[2][3] + c[2][4] + c[2][5] + c[2][6] + c[2][7];

				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];
				c[5][0] += c[5][1] + c[5][2] + c[5][3] + c[5][4] + c[5][5] + c[5][6] + c[5][7];
				c[6][0] += c[6][1] + c[6][2] + c[6][3] + c[6][4] + c[6][5] + c[6][6] + c[6][7];

				c[8][0] += c[8][1] + c[8][2] + c[8][3] + c[8][4] + c[8][5] + c[8][6] + c[8][7];
				c[9][0] += c[9][1] + c[9][2] + c[9][3] + c[9][4] + c[9][5] + c[9][6] + c[9][7];
				c[10][0] += c[10][1] + c[10][2] + c[10][3] + c[10][4] + c[10][5] + c[10][6] + c[10][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];
				mC[j * Bcolumns + i + 2] += c[2][0];

				mC[(j + 1) * Bcolumns + i] += c[4][0];
				mC[(j + 1) * Bcolumns + i + 1] += c[5][0];
				mC[(j + 1) * Bcolumns + i + 2] += c[6][0];

				mC[(j + 2) * Bcolumns + i] += c[8][0];
				mC[(j + 2) * Bcolumns + i + 1] += c[9][0];
				mC[(j + 2) * Bcolumns + i + 2] += c[10][0];


				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
					mC[j * Bcolumns + i + 2] += mA[j * columns + k] * mB[(i + 2) * columns + k];

					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
					mC[(j + 1) * Bcolumns + i + 1] += mA[(j + 1) * columns + k] * mB[(i + 1) * columns + k];
					mC[(j + 1) * Bcolumns + i + 2] += mA[(j + 1) * columns + k] * mB[(i + 2) * columns + k];

					mC[(j + 2) * Bcolumns + i] += mA[(j + 2) * columns + k] * mB[i * columns + k];
					mC[(j + 2) * Bcolumns + i + 1] += mA[(j + 2) * columns + k] * mB[(i + 1) * columns + k];
					mC[(j + 2) * Bcolumns + i + 2] += mA[(j + 2) * columns + k] * mB[(i + 2) * columns + k];
				}
			}
			if (rows - j == 2) {
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;
				C[2] = CMtx::ZERO;

				C[4] = CMtx::ZERO;
				C[5] = CMtx::ZERO;
				C[6] = CMtx::ZERO;


				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);
					B[2] = _mm256_loadu_ps(mB + (i + 2) * columns + k);


					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));
					C[2] = _mm256_add_ps(C[2], _mm256_mul_ps(A[0], B[2]));

					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));
					C[5] = _mm256_add_ps(C[5], _mm256_mul_ps(A[1], B[1]));
					C[6] = _mm256_add_ps(C[6], _mm256_mul_ps(A[1], B[2]));


				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);
				_mm256_storeu_ps(c[2], C[2]);

				_mm256_storeu_ps(c[4], C[4]);
				_mm256_storeu_ps(c[5], C[5]);
				_mm256_storeu_ps(c[6], C[6]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];
				c[2][0] += c[2][1] + c[2][2] + c[2][3] + c[2][4] + c[2][5] + c[2][6] + c[2][7];

				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];
				c[5][0] += c[5][1] + c[5][2] + c[5][3] + c[5][4] + c[5][5] + c[5][6] + c[5][7];
				c[6][0] += c[6][1] + c[6][2] + c[6][3] + c[6][4] + c[6][5] + c[6][6] + c[6][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];
				mC[j * Bcolumns + i + 2] += c[2][0];

				mC[(j + 1) * Bcolumns + i] += c[4][0];
				mC[(j + 1) * Bcolumns + i + 1] += c[5][0];
				mC[(j + 1) * Bcolumns + i + 2] += c[6][0];


				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
					mC[j * Bcolumns + i + 2] += mA[j * columns + k] * mB[(i + 2) * columns + k];

					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
					mC[(j + 1) * Bcolumns + i + 1] += mA[(j + 1) * columns + k] * mB[(i + 1) * columns + k];
					mC[(j + 1) * Bcolumns + i + 2] += mA[(j + 1) * columns + k] * mB[(i + 2) * columns + k];
				}
			}
			else if (rows - j == 1) {
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;
				C[2] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);
					B[2] = _mm256_loadu_ps(mB + (i + 2) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));
					C[2] = _mm256_add_ps(C[2], _mm256_mul_ps(A[0], B[2]));

				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);
				_mm256_storeu_ps(c[2], C[2]);


				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];
				c[2][0] += c[2][1] + c[2][2] + c[2][3] + c[2][4] + c[2][5] + c[2][6] + c[2][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];
				mC[j * Bcolumns + i + 2] += c[2][0];

				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
					mC[j * Bcolumns + i + 2] += mA[j * columns + k] * mB[(i + 2) * columns + k];
				}
			}
		}
		else if (end - i == 2) {
			j = 0;
			for (; j < rows - rows % 3; j += 3)
			{
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;

				C[4] = CMtx::ZERO;
				C[5] = CMtx::ZERO;

				C[8] = CMtx::ZERO;
				C[9] = CMtx::ZERO;


				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);
					A[2] = _mm256_loadu_ps(mA + (j + 2) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));

					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));
					C[5] = _mm256_add_ps(C[5], _mm256_mul_ps(A[1], B[1]));

					C[8] = _mm256_add_ps(C[8], _mm256_mul_ps(A[2], B[0]));
					C[9] = _mm256_add_ps(C[9], _mm256_mul_ps(A[2], B[1]));


				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);

				_mm256_storeu_ps(c[4], C[4]);
				_mm256_storeu_ps(c[5], C[5]);

				_mm256_storeu_ps(c[8], C[8]);
				_mm256_storeu_ps(c[9], C[9]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];


				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];
				c[5][0] += c[5][1] + c[5][2] + c[5][3] + c[5][4] + c[5][5] + c[5][6] + c[5][7];

				c[8][0] += c[8][1] + c[8][2] + c[8][3] + c[8][4] + c[8][5] + c[8][6] + c[8][7];
				c[9][0] += c[9][1] + c[9][2] + c[9][3] + c[9][4] + c[9][5] + c[9][6] + c[9][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];

				mC[(j + 1) * Bcolumns + i] += c[4][0];
				mC[(j + 1) * Bcolumns + i + 1] += c[5][0];

				mC[(j + 2) * Bcolumns + i] += c[8][0];
				mC[(j + 2) * Bcolumns + i + 1] += c[9][0];


				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];

					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
					mC[(j + 1) * Bcolumns + i + 1] += mA[(j + 1) * columns + k] * mB[(i + 1) * columns + k];

					mC[(j + 2) * Bcolumns + i] += mA[(j + 2) * columns + k] * mB[i * columns + k];
					mC[(j + 2) * Bcolumns + i + 1] += mA[(j + 2) * columns + k] * mB[(i + 1) * columns + k];
				}
			}
			if (rows - j == 2) {
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;

				C[4] = CMtx::ZERO;
				C[5] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));

					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));
					C[5] = _mm256_add_ps(C[5], _mm256_mul_ps(A[1], B[1]));


				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);

				_mm256_storeu_ps(c[4], C[4]);
				_mm256_storeu_ps(c[5], C[5]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];

				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];
				c[5][0] += c[5][1] + c[5][2] + c[5][3] + c[5][4] + c[5][5] + c[5][6] + c[5][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];

				mC[(j + 1) * Bcolumns + i] += c[4][0];
				mC[(j + 1) * Bcolumns + i + 1] += c[5][0];


				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];

					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
					mC[(j + 1) * Bcolumns + i + 1] += mA[(j + 1) * columns + k] * mB[(i + 1) * columns + k];
				}
			}
			else if (rows - j == 1) {
				C[0] = CMtx::ZERO;
				C[1] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);
					B[1] = _mm256_loadu_ps(mB + (i + 1) * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));
					C[1] = _mm256_add_ps(C[1], _mm256_mul_ps(A[0], B[1]));

				}
				_mm256_storeu_ps(c[0], C[0]);
				_mm256_storeu_ps(c[1], C[1]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];
				c[1][0] += c[1][1] + c[1][2] + c[1][3] + c[1][4] + c[1][5] + c[1][6] + c[1][7];

				mC[j * Bcolumns + i] += c[0][0];
				mC[j * Bcolumns + i + 1] += c[1][0];

				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
					mC[j * Bcolumns + i + 1] += mA[j * columns + k] * mB[(i + 1) * columns + k];
				}
			}
		}
		else if (end - i == 1) {
			j = 0;
			for (; j < rows - rows % 3; j += 3)
			{
				C[0] = CMtx::ZERO;

				C[4] = CMtx::ZERO;

				C[8] = CMtx::ZERO;


				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);
					A[2] = _mm256_loadu_ps(mA + (j + 2) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));

					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));

					C[8] = _mm256_add_ps(C[8], _mm256_mul_ps(A[2], B[0]));


				}
				_mm256_storeu_ps(c[0], C[0]);

				_mm256_storeu_ps(c[4], C[4]);

				_mm256_storeu_ps(c[8], C[8]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];

				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];

				c[8][0] += c[8][1] + c[8][2] + c[8][3] + c[8][4] + c[8][5] + c[8][6] + c[8][7];

				mC[j * Bcolumns + i] += c[0][0];

				mC[(j + 1) * Bcolumns + i] += c[4][0];

				mC[(j + 2) * Bcolumns + i] += c[8][0];


				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];

					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];

					mC[(j + 2) * Bcolumns + i] += mA[(j + 2) * columns + k] * mB[i * columns + k];
				}
			}
			if (rows - j == 2) {
				C[0] = CMtx::ZERO;

				C[4] = CMtx::ZERO;

				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);
					A[1] = _mm256_loadu_ps(mA + (j + 1) * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);


					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));

					C[4] = _mm256_add_ps(C[4], _mm256_mul_ps(A[1], B[0]));

				}
				_mm256_storeu_ps(c[0], C[0]);

				_mm256_storeu_ps(c[4], C[4]);

				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];

				c[4][0] += c[4][1] + c[4][2] + c[4][3] + c[4][4] + c[4][5] + c[4][6] + c[4][7];

				mC[j * Bcolumns + i] += c[0][0];

				mC[(j + 1) * Bcolumns + i] += c[4][0];


				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];

					mC[(j + 1) * Bcolumns + i] += mA[(j + 1) * columns + k] * mB[i * columns + k];
				}
			}
			else if (rows - j == 1) {
				C[0] = CMtx::ZERO;


				k = 0;
				for (; k < columns - columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
				{

					A[0] = _mm256_loadu_ps(mA + j * columns + k);

					B[0] = _mm256_loadu_ps(mB + i * columns + k);

					C[0] = _mm256_add_ps(C[0], _mm256_mul_ps(A[0], B[0]));

				}
				_mm256_storeu_ps(c[0], C[0]);


				c[0][0] += c[0][1] + c[0][2] + c[0][3] + c[0][4] + c[0][5] + c[0][6] + c[0][7];

				mC[j * Bcolumns + i] += c[0][0];

				for (; k < columns; ++k)
				{
					mC[j * Bcolumns + i] += mA[j * columns + k] * mB[i * columns + k];
				}
			}
		}

		delete[] c[0];
		delete[] c[1];
		delete[] c[2];
		delete[] c[3];
		delete[] c[4];
		delete[] c[5];
		delete[] c[6];
		delete[] c[7];
		delete[] c[8];
		delete[] c[9];
		delete[] c[10];
		delete[] c[11];

	}

	inline float* CMtx::multArray(const CMtx& rhs) const& throw()
	{
		if (columns != rhs.rows)
			throw std::invalid_argument("Can't multiply matrices with that sizes");

		float* matrixA = matrix;
		float* matrixC = CMtx::zeroArray(rows, rhs.columns);
		int start, end;

		if (columns > CMtx::THREAD_BORDER || (rows > CMtx::THREAD_BORDER && rhs.columns > CMtx::THREAD_BORDER)) {
			float* matrixB = rhs.transposeArray();
			if (rhs.columns >= CMtx::THREAD_BORDER) {

				std::thread myThreads[CMtx::THREAD_NUMBER];

				for (int threadNumber = 0; threadNumber < CMtx::THREAD_NUMBER - 1; ++threadNumber) {
					start = threadNumber * rhs.columns / CMtx::THREAD_NUMBER;
					end = (threadNumber + 1) * rhs.columns / CMtx::THREAD_NUMBER;
					myThreads[threadNumber] = std::thread(&CMtx::multThread, &*this, start, end, std::ref(matrixA), std::ref(matrixB), std::ref(matrixC), rhs.columns);
				}
				start = (CMtx::THREAD_NUMBER - 1) * rhs.columns / CMtx::THREAD_NUMBER;
				end = rhs.columns;
				myThreads[CMtx::THREAD_NUMBER - 1] = std::thread(&CMtx::multThread, &*this, start, end, std::ref(matrixA), std::ref(matrixB), std::ref(matrixC), rhs.columns);

				for (int threadNumber = 0; threadNumber < CMtx::THREAD_NUMBER; ++threadNumber)
					myThreads[threadNumber].join();
			}
			else {
				multThread(0, rhs.columns, matrixA, matrixB, matrixC, rhs.columns);
			}
			delete[] matrixB;
		}
		else
		{
			__m256 mult;
			int k, l;
			for (int i = 0; i < rows; ++i) {

				k = 0;
				for (; k < rhs.columns - rhs.columns % CMtx::SIMD_LENGTH; k += CMtx::SIMD_LENGTH)
					_mm256_storeu_ps(matrixC + i * rhs.columns + k, CMtx::ZERO);
				for (; k < rhs.columns; ++k)
					matrixC[i * rhs.columns + k] = 0.0f;

				for (int j = 0; j < columns; ++j) {
					mult = _mm256_set1_ps(matrix[i * columns + j]);
					l = 0;
					for (; l < rhs.columns - rhs.columns % CMtx::SIMD_LENGTH; l += CMtx::SIMD_LENGTH)
						_mm256_storeu_ps(matrixC + i * rhs.columns + l, _mm256_add_ps(_mm256_loadu_ps(matrixC + i * rhs.columns + l), _mm256_mul_ps(_mm256_loadu_ps(rhs.matrix + j * rhs.columns + l), mult)));
					for (; l < rhs.columns; ++l)
						matrixC[i * rhs.columns + l] += rhs.matrix[j * rhs.columns + l] * matrix[i * columns + j];
				}
			}
		}

		return matrixC;
	}

	CMtx CMtx::operator*(const CMtx& rhs) const& throw()
	{
		float* m = multArray(rhs);
		return CMtx(rows, rhs.columns, m);
	}

	CMtx CMtx::operator*(CMtx&& rhs) const& throw()
	{
		rhs.matrix = multArray(rhs);
		rhs.rows = rows;
		return rhs;
	}

	CMtx CMtx::operator*(const CMtx& rhs) && throw()
	{
		matrix = multArray(rhs);
		columns = rhs.columns;
		return std::move(*this);
	}

	CMtx CMtx::operator*(CMtx&& rhs) && throw()
	{
		rhs.matrix = multArray(rhs);
		rhs.rows = rows;
		return rhs;
	}

	CMtx CMtx::operator*(float multiplier) const& throw()
	{
		float* m = scalarMult(multiplier);
		return CMtx(rows, columns, m);
	}

	CMtx CMtx::operator*(float multiplier) && throw()
	{
		scalarMultMove(multiplier);
		return std::move(*this);
	}

	float* CMtx::scalarMult(float multiplier) const
	{
		float* m = new float[rows * columns];
		__m256 mult = _mm256_set1_ps(multiplier);

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], _mm256_mul_ps(_mm256_loadu_ps(&matrix[i]), mult));
		for (; i < rows * columns; ++i)
			m[i] = matrix[i] * multiplier;
		return m;
	}

	void CMtx::scalarMultMove(float multiplier)
	{
		__m256 mult = _mm256_set1_ps(multiplier);

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&matrix[i], _mm256_mul_ps(_mm256_loadu_ps(&matrix[i]), mult));
		for (; i < rows * columns; ++i)
			matrix[i] = matrix[i] * multiplier;
	}

	CMtx CMtx::operator+(const CMtx& rhs) const& throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't add matrices with different sizes");
		float* m = new float[rows * columns];

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], _mm256_add_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			m[i] = matrix[i] + rhs.matrix[i];

		return CMtx(rows, columns, m);
	}

	CMtx CMtx::operator+(const CMtx& rhs) && throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't add matrices with different sizes");

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&matrix[i], _mm256_add_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			matrix[i] = matrix[i] + rhs.matrix[i];

		return std::move(*this);
	}

	CMtx CMtx::operator+(CMtx&& rhs) const& throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't add matrices with different sizes");

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&rhs.matrix[i], _mm256_add_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			rhs.matrix[i] = matrix[i] + rhs.matrix[i];

		return rhs;
	}

	CMtx CMtx::operator+(CMtx&& rhs) && throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't add matrices with different sizes");

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&rhs.matrix[i], _mm256_add_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			rhs.matrix[i] = matrix[i] + rhs.matrix[i];

		return rhs;
	}

	CMtx CMtx::operator-(const CMtx& rhs) const& throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't subtract matrices with different sizes");
		float* m = new float[rows * columns];

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], _mm256_sub_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			m[i] = matrix[i] - rhs.matrix[i];

		return CMtx(rows, columns, m);
	}

	CMtx CMtx::operator-(const CMtx& rhs) && throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't subtract matrices with different sizes");

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&matrix[i], _mm256_sub_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			matrix[i] = matrix[i] - rhs.matrix[i];

		return std::move(*this);
	}

	CMtx CMtx::operator-(CMtx&& rhs) const& throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't subtract matrices with different sizes");

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&rhs.matrix[i], _mm256_sub_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			rhs.matrix[i] = matrix[i] - rhs.matrix[i];

		return rhs;
	}

	CMtx CMtx::operator-(CMtx&& rhs) && throw()
	{
		if (rows != rhs.rows || columns != rhs.columns)
			throw std::invalid_argument("Can't subtract matrices with different sizes");

		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&rhs.matrix[i], _mm256_sub_ps(_mm256_loadu_ps(&matrix[i]), _mm256_loadu_ps(&rhs.matrix[i])));
		for (; i < rows * columns; ++i)
			rhs.matrix[i] = matrix[i] - rhs.matrix[i];

		return rhs;
	}

	CMtx CMtx::operator-() const&
	{
		float* m = new float[rows * columns];
		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], _mm256_sub_ps(CMtx::ZERO, _mm256_loadu_ps(&matrix[i])));
		for (; i < rows * columns; ++i)
			m[i] = -matrix[i];

		return CMtx(rows, columns, m);
	}

	CMtx CMtx::operator-()&&
	{
		int i = 0;
		for (; i < rows * columns - rows * columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&matrix[i], _mm256_sub_ps(CMtx::ZERO, _mm256_loadu_ps(&matrix[i])));
		for (; i < rows * columns; ++i)
			matrix[i] = -matrix[i];

		return std::move(*this);
	}

	void CMtx::transThread(int start, int end, float*& m) const
	{
		for (int i = start; i < end; ++i)
			for (int j = 0; j < columns; ++j)
				m[j * rows + i] = matrix[i * columns + j];
	}

	inline float* CMtx::transposeArray() const
	{
		float* m = new float[rows * columns];

		if (rows >= CMtx::THREAD_BORDER) {
			std::thread myThreads[CMtx::THREAD_NUMBER];

			for (int threadNumber = 0; threadNumber < CMtx::THREAD_NUMBER - 1; ++threadNumber)
				myThreads[threadNumber] = std::thread(&CMtx::transThread, &*this, threadNumber * rows / CMtx::THREAD_NUMBER, (threadNumber + 1) * rows / CMtx::THREAD_NUMBER, std::ref(m));

			myThreads[CMtx::THREAD_NUMBER - 1] = std::thread(&CMtx::transThread, &*this, (CMtx::THREAD_NUMBER - 1) * rows / CMtx::THREAD_NUMBER, rows, std::ref(m));

			for (int threadNumber = 0; threadNumber < CMtx::THREAD_NUMBER; ++threadNumber)
				myThreads[threadNumber].join();
		}
		else
			transThread(0, rows, m);

		return m;
	}

	CMtx CMtx::operator~() const&
	{
		float* m = transposeArray();
		return CMtx(columns, rows, m);
	}

	CMtx CMtx::operator~()&&
	{
		float* m = transposeArray();
		delete[] matrix;
		matrix = m;
		return std::move(*this);
	}

	CMtx CMtx::operator^(int power) const throw()
	{
		if (rows != columns)
			throw std::exception("Can't use ^ on non-square matrix");
		if (power < 0)
			throw std::exception("Power can't be less than 0 ");
		else if (power == 0)
			return CMtx(rows, (float)1.0f);
		else if (power == 1)
			return *this;
		else {
			CMtx tmp = *this;
			for (int i = 0; i < power - 1; ++i)
				tmp = std::move(tmp * (*this));
			return tmp;
		}
	}

	bool CMtx::operator==(const CMtx& rhs) const
	{
		if (rows != rhs.rows || columns != rhs.columns)
			return false;
		float tmp;
		for (int i = 0; i < rows * columns; ++i) {
			tmp = matrix[i] - rhs.matrix[i];
			if (tmp > CMtx::ALG_PRECISION || tmp < -CMtx::ALG_PRECISION)
				return false;
		}

		return true;
	}

	const CMtx CMtx::getHorizontalVector(int id) const throw()
	{
		if (id >= rows || id < 0)
			throw std::invalid_argument("Index out of bonds of matrix: " + id);
		float* m = new float[columns];

		int i = 0;
		for (; i < columns - columns % CMtx::SIMD_LENGTH; i += CMtx::SIMD_LENGTH)
			_mm256_storeu_ps(&m[i], _mm256_loadu_ps(&matrix[i + id * columns]));
		for (; i < columns; ++i)
			m[i] = matrix[i + id * columns];

		return CMtx(1, columns, m);
	}

	const CMtx CMtx::getVerticalVector(int id) const throw()
	{
		if (id >= columns || id < 0)
			throw std::invalid_argument("Index out of bonds of matrix: " + id);
		float* m = new float[rows];
		for (int i = 0; i < rows; ++i)
			m[i] = matrix[i * rows + id];
		return CMtx(rows, 1, m);
	}

	void CMtx::display() const
	{
		for (int i = 0; i < rows; ++i) {
			std::cout << "| ";
			for (int j = 0; j < columns; ++j)
				std::cout << matrix[i * columns + j] << " ";
			std::cout << "|" << std::endl;
		}
		std::cout << std::endl;
	}


	CMtx operator*(float multiplier, const CMtx& rhs)
	{
		float* m = rhs.scalarMult(multiplier);
		return CMtx(rhs.rows, rhs.columns, m);
	}

	CMtx operator*(float multiplier, CMtx&& rhs)
	{
		rhs.scalarMultMove(multiplier);
		return rhs;
	}

}