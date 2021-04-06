#pragma once
#include <string>
#include <immintrin.h>

namespace MyAlgebra {

	class CMtx
	{
	private:
		float* matrix;
		int rows, columns;

		CMtx(int row_cnt, int col_cnt, float*& matrix_cnt);

		void multThread(int start, int end, float*& mA, float*& mB, float*& mC, int Bcolumns) const;
		void transThread(int start, int end, float*& m) const;

		inline float* copyArray() const;
		inline float* transposeArray() const;

		inline float* scalarMult(float multiplier) const;
		inline void scalarMultMove(float multiplier);
		inline float* multArray(const CMtx& rhs) const&;

		static inline float* zeroArray(int rows, int columns);
		static const float ALG_PRECISION;
		static const std::string AUTHOR;
		static const __m256 ZERO;
		static const int SIMD_LENGTH;
		static const int THREAD_NUMBER;
		static const int THREAD_BORDER;

	public:
		std::string authorName();

		CMtx(int row_cnt, int col_cnt, bool rand_init = false);

		CMtx(int row_cnt, float diagonal);

		CMtx(const CMtx& rhs);

		CMtx(CMtx&& rhs);

		~CMtx();

		int getColumns();

		int getRows();

		const CMtx& operator=(const CMtx& rhs);

		const CMtx& operator=(CMtx&& rhs);

		const CMtx& operator=(float diagonal);

		float* operator[](int id);

		CMtx operator*(const CMtx& rhs) const&;
		CMtx operator*(CMtx&& rhs) const&;
		CMtx operator*(const CMtx& rhs)&&;
		CMtx operator*(CMtx&& rhs)&&;

		CMtx operator*(float multiplier) const&;
		CMtx operator*(float multiplier)&&;

		CMtx operator+(const CMtx& rhs) const&;
		CMtx operator+(CMtx&& rhs) const&;
		CMtx operator+(const CMtx& rhs)&&;
		CMtx operator+(CMtx&& rhs)&&;

		CMtx operator-(const CMtx& rhs) const&;
		CMtx operator-(CMtx&& rhs) const&;
		CMtx operator-(const CMtx& rhs)&&;
		CMtx operator-(CMtx&& rhs)&&;

		CMtx operator-() const&; //Zamiana znaku
		CMtx operator-() &&;

		CMtx operator~() const&; //Transponowanie
		CMtx operator~() &&;

		// Akceptuje tylko power >= 0:
		//    power = 0  - zwraca macierz jednostkow¹
		//    power = 1  - zwraca kopiê macierzy
		//    power > 1  - zwraca iloczyn macierzy 
		CMtx operator^(int power) const;

		bool operator==(const CMtx& rhs) const;

		const CMtx getHorizontalVector(int id) const;

		const CMtx getVerticalVector(int id) const;

		void display() const;

		friend CMtx operator*(float multiplier, const CMtx& rhs);

		friend CMtx operator*(float multiplier, CMtx&& rhs);
	};

	CMtx operator*(float multiplier, const CMtx& rhs);

	CMtx operator*(float multiplier, CMtx&& rhs);


}




