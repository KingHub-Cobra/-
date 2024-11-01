#include <stdio.h>
#include <math.h>
#include <omp.h>

#define M 80
#define N 90

#define delta 1e-7

double left_top(double x, double y) {
    return -y+2+x;
}

double right_top(double x, double y) {
    return -y+2-x;
}

double left_bottom(double x, double y) {
    return y+2+x;
}

double right_bottom(double x, double y) {
    return y+2-x;
}

double calc_line_v(double y_t, double y_b, double x, double EPS) {
    double y_temp_1, y_temp_2;

    if (left_top(x, y_b) <= 0 || right_top(x, y_b) <= 0 || left_bottom(x, y_t) <= 0 || right_bottom(x, y_t) <= 0) {
        return 1.0 / EPS * (y_t - y_b);
    // все точки внутри фигуры
    } else if (left_top(x, y_t) >= 0 && right_top(x, y_t) >= 0 && left_bottom(x, y_b) >= 0 && right_bottom(x, y_b) >= 0) {
        return y_t - y_b;
    } else if (left_top(x, y_t) < 0 && left_bottom(x, y_b) < 0) {

        y_temp_1 = 2 + x;
        y_temp_2 = -2 - x;

        return 1.0 / EPS * (y_t - y_temp_1 + y_temp_2 - y_b) + y_temp_1 - y_temp_2;
    } else if (right_top(x, y_t) < 0 && right_bottom(x, y_b) < 0) {

        y_temp_1 = 2 - x;
        y_temp_2 = -2 + x;

        return 1.0 / EPS * (y_t - y_temp_1 + y_temp_2 - y_b) + y_temp_1 - y_temp_2;
    } else if (left_top(x, y_t) < 0) {

        y_temp_1 = 2 + x;
        return 1.0 / EPS * (y_t - y_temp_1) + y_temp_1 - y_b;
    } else if (right_top(x, y_t) < 0) {

        y_temp_1 = 2 - x;
        return 1.0 / EPS * (y_t - y_temp_1) + y_temp_1 - y_b;
    } else if (left_bottom(x, y_b) < 0) {

        y_temp_1 = -2 - x;
        return 1.0 / EPS * (y_temp_1 - y_b) + y_t - y_temp_1;
    } else if (right_bottom(x, y_b) < 0) {

        y_temp_1 = -2 + x;
        return 1.0 / EPS * (y_temp_1 - y_b) + y_t - y_temp_1;
    }
}

double calc_line_h(double x_l, double x_r, double y, double EPS) {
    double x_temp_1, x_temp_2;

    if (left_top(x_r, y) <= 0 || right_top(x_l, y) <= 0 || left_bottom(x_r, y) <= 0 || right_bottom(x_l, y) <= 0) {
        return 1.0 / EPS * (x_r - x_l);
    // все точки внутри фигуры
    } else if (left_top(x_l, y) >= 0 && right_top(x_r, y) >= 0 && left_bottom(x_l, y) >= 0 && right_bottom(x_r, y) >= 0) {
        return x_r - x_l;
    } else if (left_bottom(x_l, y) < 0 && right_bottom(x_r, y) < 0) {

        x_temp_1 = -2 - y;
        x_temp_2 = 2 + y;

        return 1.0 / EPS * (x_r - x_temp_2 + x_temp_1 - x_l) + x_temp_2 - x_temp_1;
    } else if (left_top(x_l, y) < 0) {

        x_temp_1 = y - 2;
        return 1.0 / EPS * (x_temp_1 - x_l) + x_r - x_temp_1;
    } else if (right_top(x_r, y) < 0) {

        x_temp_1 = 2 - y;
        return 1.0 / EPS * (x_r - x_temp_1) + x_temp_1 - x_l;
    } else if (left_bottom(x_l, y) < 0) {

        x_temp_1 = -2 - y;
        return 1.0 / EPS * (x_temp_1 - x_l) + x_r - x_temp_1;
    } else if (right_bottom(x_r, y) < 0) {

        x_temp_1 = 2 + y;
        return 1.0 / EPS * (x_r - x_temp_1) + x_temp_1 - x_l;
    }
}

double calc_area(double x_l, double x_r, double y_b, double y_t) {
    double x_temp_1, y_temp_1, x_temp_2, y_temp_2, x_temp_3, y_temp_3, x_temp_4, y_temp_4;
    // все точки вне фигуры
    if (left_top(x_r, y_b) <= 0 || right_top(x_l, y_b) <= 0 || left_bottom(x_r, y_t) <= 0 || right_bottom(x_l, y_t) <= 0) {
        //printf("IUGHJG\n");
        return 0.0;
    // все точки внутри фигуры
    } else if (left_top(x_l, y_t) >= 0 && right_top(x_r, y_t) >= 0 && left_bottom(x_l, y_b) >= 0 && right_bottom(x_r, y_b) >= 0) {
        return (x_r-x_l)*(y_t-y_b);
    // точки лежат по-разному
    } else if (left_top(x_l, y_t) < 0 && left_bottom(x_l, y_b) < 0) { // левый прямоугольник, пересечение 2 и 3 четверти
        x_temp_2 = x_l;
        y_temp_2 = 2 + x_l;

        x_temp_3 = x_l;
        y_temp_3 = -2 - x_l;
        if (left_top(x_r, y_t) < 0 && left_bottom(x_r, y_b) < 0) { // отсекаются трапеции
            x_temp_1 = x_r;
            y_temp_1 = 2 + x_r;
            
            x_temp_4 = x_r;
            y_temp_4 = -2 - x_r;

            return (x_r-x_l)*(y_t-y_b) - (y_t-y_temp_2 + y_t-y_temp_1)*(x_r-x_l)/2 - (y_temp_3-y_b + y_temp_4-y_b)*(x_r-x_l)/2;
        } else { // отсекаются треугольники
            x_temp_1 = y_t - 2;
            y_temp_1 = y_t;

            x_temp_4 = -2 - y_b;
            y_temp_4 = y_b;

            return (x_r-x_l)*(y_t-y_b) - (y_t-y_temp_2)*(x_temp_1-x_l)/2 - (y_temp_3-y_b)*(x_temp_4-x_l)/2;
        }
    } else if (right_top(x_r, y_t) < 0 && right_bottom(x_r, y_b) < 0) { // правый прямоугольник, пересечение 1 и 4 четверти
        x_temp_2 = x_r;
        y_temp_2 = 2 - x_r;

        x_temp_3 = x_r;
        y_temp_3 = -2 + x_r;
        if (right_top(x_l, y_t) < 0 && right_bottom(x_l, y_b) < 0) { // отсекаются трапеции
            x_temp_1 = x_l;
            y_temp_1 = 2 - x_l;
            
            x_temp_4 = x_l;
            y_temp_4 = -2 + x_l;

            return (x_r-x_l)*(y_t-y_b) - (y_t-y_temp_2 + y_t-y_temp_1)*(x_r-x_l)/2 - (y_temp_3-y_b + y_temp_4-y_b)*(x_r-x_l)/2;
        } else { // отсекаются треугольники
            x_temp_1 = 2 - y_t;
            y_temp_1 = y_t;

            x_temp_4 = 2 + y_b;
            y_temp_4 = y_b;

            return (x_r-x_l)*(y_t-y_b) - (y_t-y_temp_2)*(x_r - x_temp_1)/2 - (y_temp_3-y_b)*(x_r - x_temp_4)/2;
        }
    } else if (left_bottom(x_l, y_b) < 0 && right_bottom(x_r, y_b) < 0) { // нижний прямоугольник, пересечение 3 и 4 четверти
        x_temp_2 = -2 - y_b;
        y_temp_2 = y_b;

        x_temp_3 = y_b + 2;
        y_temp_3 = y_b;
        if (left_bottom(x_l, y_t) < 0 && right_bottom(x_r, y_t) < 0) { // отсекаются трапеции
            x_temp_1 = -2 - y_t;
            y_temp_1 = y_t;
            
            x_temp_4 = y_t + 2;
            y_temp_4 = y_t;

            return (x_r-x_l)*(y_t-y_b) - (y_t-y_b)*(x_temp_1-x_l + x_temp_2-x_l)/2 - (y_t-y_b)*(x_r-x_temp_4 + x_r-x_temp_3)/2;
        } else { // отсекаются треугольники
            x_temp_1 = x_l;
            y_temp_1 = -2 - x_l;

            x_temp_4 = x_r;
            y_temp_4 = -2 + x_r;

            return (x_r-x_l)*(y_t-y_b) - (y_temp_1-y_b)*(x_temp_2 - x_l)/2 - (y_temp_4-y_b)*(x_r - x_temp_3)/2;
        }
    } else if (left_top(x_l, y_t) < 0) { // точка за фигурой во 2 четверти
        if (left_top(x_l, y_b) < 0 && left_top(x_r, y_t) < 0) { // отсекается пятиугольник
            x_temp_1 = x_r;
            y_temp_1 = 2 + x_r;

            x_temp_2 = y_b - 2;
            y_temp_2 = y_b;
            return (x_temp_1 - x_temp_2)*(y_temp_1 - y_temp_2)/2;
        } else if (left_top(x_l, y_b) < 0) {
            x_temp_1 = y_t - 2;
            y_temp_1 = y_t;

            x_temp_2 = y_b - 2;
            y_temp_2 = y_b;
            return (x_r-x_temp_1 + x_r-x_temp_2)*(y_t-y_b)/2;
        } else if (left_top(x_r, y_t) < 0) {
            x_temp_1 = x_r;
            y_temp_1 = 2 + x_r;

            x_temp_2 = x_l;
            y_temp_2 = 2 + x_l;
            return (x_r-x_l)*(y_temp_1-y_b + y_temp_2-y_b)/2;
        } else { // отсекается треугольник
            x_temp_1 = y_t - 2;
            y_temp_1 = y_t;

            x_temp_2 = x_l;
            y_temp_2 = 2 + x_l;
            return (x_r-x_l)*(y_t-y_b) - (x_temp_1 - x_l)*(y_t - y_temp_2)/2;
        }
    } else if (right_top(x_r, y_t) < 0) { // точка за фигурой в 1 четверти
        if (right_top(x_l, y_t) < 0 && right_top(x_r, y_b) < 0) { // отсекается пятиугольник
            x_temp_1 = x_l;
            y_temp_1 = 2 - x_l;

            x_temp_2 = 2 - y_b;
            y_temp_2 = y_b;
            return (x_temp_2 - x_l)*(y_temp_1 - y_b)/2;
        } else if (right_top(x_r, y_b) < 0) {
            x_temp_1 = -y_t + 2;
            y_temp_1 = y_t;

            x_temp_2 = -y_b + 2;
            y_temp_2 = y_b;
            return (x_temp_1-x_l + x_temp_2-x_l)*(y_t-y_b)/2;
        } else if (right_top(x_l, y_t) < 0) {
            x_temp_1 = x_l;
            y_temp_1 = 2 - x_l;

            x_temp_2 = x_r;
            y_temp_2 = 2 - x_r;
            return (x_r-x_l)*(y_temp_1-y_b + y_temp_2-y_b)/2;
        } else { // отсекается треугольник
            x_temp_1 = -y_t + 2;
            y_temp_1 = y_t;

            x_temp_2 = x_r;
            y_temp_2 = 2 - x_r;
            return (x_r-x_l)*(y_t-y_b) - (x_r - x_temp_1)*(y_t - y_temp_2)/2;
        }
    } else if (left_bottom(x_l, y_b) < 0) { // точка за фигурой в 3 четверти
        if (left_bottom(x_l, y_t) < 0 && left_bottom(x_r, y_b) < 0) { // отсекается пятиугольник
            x_temp_1 = -2 - y_t;
            y_temp_1 = y_t;

            x_temp_2 = x_r;
            y_temp_2 = -2 - x_r;
            return (x_r - x_temp_1)*(y_t - y_temp_2)/2;
        } else if (left_bottom(x_l, y_t) < 0) {
            x_temp_1 = -y_t - 2;
            y_temp_1 = y_t;

            x_temp_2 = -y_b - 2;
            y_temp_2 = y_b;
            return (x_r-x_temp_1 + x_r-x_temp_2)*(y_t-y_b)/2;
        } else if (left_bottom(x_r, y_b) < 0) {
            x_temp_1 = x_l;
            y_temp_1 = -2 - x_l;

            x_temp_2 = x_r;
            y_temp_2 = -2 - x_r;
            return (x_r-x_l)*(y_t-y_temp_1 + y_t-y_temp_2)/2;
        } else { // отсекается треугольник
            x_temp_1 = x_l;
            y_temp_1 = -2 - x_l;

            x_temp_2 = -2 - y_b;
            y_temp_2 = y_b;
            return (x_r-x_l)*(y_t-y_b) - (y_temp_1-y_b)*(x_temp_2-x_l)/2;
        }
    } else { // точка за фигурой в 4 четверти
        if (right_bottom(x_r, y_t) < 0 && right_bottom(x_l, y_b) < 0) { // отсекается пятиугольник
            x_temp_1 = 2 + y_t;
            y_temp_1 = y_t;

            x_temp_2 = x_l;
            y_temp_2 = -2 + x_l;
            return (x_temp_1 - x_l)*(y_t - y_temp_2)/2;
        } else if (right_bottom(x_r, y_t) < 0) {
            x_temp_1 = y_t + 2;
            y_temp_1 = y_t;

            x_temp_2 = y_b + 2;
            y_temp_2 = y_b;
            return (x_temp_1-x_l + x_temp_2-x_l)*(y_t-y_b)/2;
        } else if (right_bottom(x_l, y_b) < 0) {
            x_temp_1 = x_r;
            y_temp_1 = -2 + x_r;

            x_temp_2 = x_l;
            y_temp_2 = -2 + x_l;
            return (x_r-x_l)*(y_t-y_temp_1 + y_t-y_temp_2)/2;
        } else { // отсекается треугольник
            x_temp_1 = x_r;
            y_temp_1 = -2 + x_r;

            x_temp_2 = 2 + y_b;
            y_temp_2 = y_b;
            return (x_r-x_l)*(y_t-y_b) - (y_temp_1-y_b)*(x_r-x_temp_2)/2;
        }
    }
}

int main() {

    omp_set_num_threads(32);

    double start_time = omp_get_wtime(), end_time;

    int i, j;

    
    double a[M][N], b[M][N], F[M][N], w[M][N], r[M][N], Ar[M][N], w_rotated[N][M];

    // #pragma omp parallel for collapse(2) private(i, j) shared(a, b, F, r, w, Ar, w_rotated)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = 0.0;
            b[i][j] = 0.0;
            F[i][j] = 0.0;
            w[i][j] = 0.0;
            r[i][j] = 0.0;
            Ar[i][j] = 0.0;
            w_rotated[i][j] = 0.0;
        }
    }

    double h_1 = 4.0/M, h_2 = 3.0/N;
    double EPS = fmax(h_1, h_2)*fmax(h_1, h_2) ;
    // printf("%lf %lf", h_1, h_2);
    double x_i, y_j;
    double A_1 = -2.0, B_1 = 2.0, A_2 = -2.0, B_2 = 1.0;

    // #pragma omp parallel for collapse(2) private(i, j, x_i, y_j) shared(a, b, F)
    for (i = 1; i < M; i++) {
        for (j = 1; j < N; j++) {
            x_i = A_1 + i*h_1;
            y_j = A_2 + j*h_2;

            //printf("%lf %lf", x_i, y_j);
            if (i != M - 1 && j != N - 1) {
                F[i][j] = 1.0 / (h_1 * h_2) * calc_area(x_i - 0.5*h_1, x_i + 0.5*h_1, y_j - 0.5*h_2, y_j + 0.5*h_2);
            }
            a[i][j] = 1.0 / h_2 * calc_line_v(y_j + 0.5*h_2, y_j - 0.5*h_2, x_i - 0.5*h_1, EPS);
            b[i][j] = 1.0 / h_1 * calc_line_h(x_i - 0.5*h_1, x_i + 0.5*h_1, y_j - 0.5*h_2, EPS);
        }
    }

    double scalar_top = 0.0, scalar_bottom = 0.0, tau = 0.0, norm = 1.0; 
    int iter_cnt = 0, step = 100000, ii = 0;
    double mas[33];
    for (i = 0; i < 33; i++) {
        mas[i] = 0;
    }
    while (norm > delta)
    {
        iter_cnt++;
        norm = 0.0;

        #pragma omp parallel for collapse(2) reduction(+:scalar_top) private(i, j) shared(a, b, F, r, w)
        for (i = 1; i < M - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                r[i][j] = -(a[i+1][j] * (w[i+1][j] - w[i][j]) / h_1 - a[i][j] * (w[i][j] - w[i-1][j]) / h_1) / h_1
                            - (b[i][j+1] * (w[i][j+1] - w[i][j]) / h_2 - b[i][j] * (w[i][j] - w[i][j-1]) / h_2) / h_2
                            - F[i][j];
                scalar_top += r[i][j] * r[i][j];
            }
        }
        scalar_top *= h_1 * h_2;
        if (iter_cnt % step == 0) {
            mas[ii] = sqrt(scalar_top);
            ii++;
        }

        #pragma omp parallel for collapse(2) reduction(+:scalar_bottom) private(i, j) shared(a, b, F, r, w, Ar)
        for (i = 1; i < M - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                Ar[i][j] = -(a[i+1][j] * (r[i+1][j] - r[i][j]) / h_1 - a[i][j] * (r[i][j] - r[i-1][j]) / h_1) / h_1
                            - (b[i][j+1] * (r[i][j+1] - r[i][j]) / h_2 - b[i][j] * (r[i][j] - r[i][j-1]) / h_2) / h_2;
                scalar_bottom += Ar[i][j] * r[i][j];
            }
        }
        scalar_bottom *= h_1 * h_2;

        tau = scalar_top / scalar_bottom;
        
        #pragma omp parallel for collapse(2) reduction(+:norm) private(i, j) shared(a, b, F, r, w, w_rotated)
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                double prev_value = w[i][j];
                w[i][j] = prev_value - tau * r[i][j];

                w_rotated[j][i] = w[i][M-j-1];

                norm += (w[i][j] - prev_value) * (w[i][j] - prev_value);
            }
        }

        norm *= h_1 * h_2;
        norm = sqrt(norm);

        scalar_top = 0.0;
        scalar_bottom = 0.0;
    }

    // for (i = 0; i < 33; i++) {
    //     printf("%lf, ", mas[i]);
    // }
    // for (i = 0; i < N; i++) {
    //     for (j = 0; j < M; j++) {
    //         printf("%lf ", w_rotated[i][j]);
    //     }
    //     printf("\n");
    // }

    printf("Programm finished for %lf seconds\n", omp_get_wtime() - start_time);
    printf("Iterations count: %d", iter_cnt);

    return 0;
}