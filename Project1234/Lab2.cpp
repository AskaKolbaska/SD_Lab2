#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cblas.h>

using namespace std;


float** new_matrix(const int N) {
    float** a = new float* [N];
    float* memp = new float[N * N];
    for (int i = 0; i < N; i++) {
        a[i] = memp + i * N;
    }
    return a;
}

void matrix_multiply(float** A, float** B, float** C, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float r = A[i][k];
            for (int j = 0; j < N; ++j) {
                C[i][j] += r * B[k][j];
            }
        }
    }
}


void block_matrix_multiply(float** A, float** B, float** C, int N, int blockSize) {
#pragma omp parallel for
    for (int i = 0; i < N; i += blockSize) {
        for (int k = 0; k < N; k += blockSize) {
            for (int j = 0; j < N; j += blockSize) {
                for (int ii = i; ii < min(i + blockSize, N); ++ii) {
                    for (int kk = k; kk < min(k + blockSize, N); ++kk) {
                        float r = A[ii][kk];
                        for (int jj = j; jj < min(j + blockSize, N); ++jj) {
                            C[ii][jj] += r * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}


bool compare_matrices(float** A, float** B, int rows, int cols, float epsilon = 1e-2f) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fabs(A[i][j] - B[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    setlocale(LC_ALL, "Russian");
    const int N = 2048;
    int blockSize = 16;
    float** a = new_matrix(N);
    float** b = new_matrix(N);
    float** c1 = new_matrix(N);
    float** c2 = new_matrix(N);
    float** c3 = new_matrix(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = static_cast<float>(rand()) / RAND_MAX;
            b[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Первый метод: Простое умножение матриц
    clock_t start = clock();
    matrix_multiply(a, b, c1, N);
    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "Простое умножение: Время выполнения = " << elapsed_secs << " секунд." << endl;
    cout << "Производительность = " << (2.0 * N * N * N / elapsed_secs * 1.0e-6) << " MFlops." << endl;

    // Второй метод: OpenBLAS SGEMM
    float alpha = 1.0f, beta = 0.0f;
    start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, &a[0][0], N, &b[0][0], N, beta, &c2[0][0], N);
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "OpenBLAS SGEMM: Время выполнения = " << elapsed_secs << " секунд." << endl;
    cout << "Производительность = " << (2.0 * N * N * N / elapsed_secs * 1.0e-6) << " MFlops." << endl;

  
    // Третий метод: Блочное умножение матриц
    start = clock();
    block_matrix_multiply(a, b, c3, N, blockSize);
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "Блочное умножение: Время выполнения = " << elapsed_secs << " секунд." << endl;
    cout << "Производительность = " << (2.0 * N * N * N / elapsed_secs * 1.0e-6) << " MFlops." << endl;
  
    // Сравнение матриц для проверки корректности
    if (compare_matrices(c1, c2, N, N))
        cout << "Матрицы c1 и c2 равны." << endl;
    else
        cout << "Матрицы c1 и c2 не равны!" << endl;

    if (compare_matrices(c1, c3, N, N))
        cout << "Матрицы c1 и c3 равны." << endl;
    else
        cout << "Матрицы c1 и c3 не равны!" << endl;

    cout << "a[10][20] = " << a[10][20] << "\n";

    cout << "b[20][10] = " << b[20][10] << "\n";

    return 0;
}