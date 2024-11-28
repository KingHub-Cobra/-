#include <stdio.h>
#include <stdlib.h> //free
#include <math.h>
#include <mpi.h>

#define M 40
#define N 40

#define delta 1e-6

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
        //// // printf("IUGHJG\n");
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



int main(int argc, char *argv[]) {

    // printf("Main started");

    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int k = size / 2 + size % 2;
    int n = size / k;
    int tup_x[2] = {rank / k, rank % k};

    int row_start = (N/n - 1) * tup_x[0];
    int col_start = (N/k - 1) * tup_x[1];
    int row_end = N/n - 1 + tup_x[0]*(N/n) + (n-1)*(!tup_x[0]);
    int col_end = N/k - 1 + tup_x[1]*(N/k) + (k-1)*(!tup_x[1]);

    int i, j;

    double a[M][N], b[M][N], F[M][N], w[M][N], r[M][N], Ar[M][N], w_rotated[N][M];

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
    double x_i, y_j;
    double A_1 = -2.0, B_1 = 2.0, A_2 = -2.0, B_2 = 1.0;

    for (i = 1; i < M; i++) {
        for (j = 1; j < N; j++) {
            x_i = A_1 + i*h_1;
            y_j = A_2 + j*h_2;

            if (i != M - 1 && j != N - 1) {
                F[i][j] = 1.0 / (h_1 * h_2) * calc_area(x_i - 0.5*h_1, x_i + 0.5*h_1, y_j - 0.5*h_2, y_j + 0.5*h_2);
            }
            a[i][j] = 1.0 / h_2 * calc_line_v(y_j + 0.5*h_2, y_j - 0.5*h_2, x_i - 0.5*h_1, EPS);
            b[i][j] = 1.0 / h_1 * calc_line_h(x_i - 0.5*h_1, x_i + 0.5*h_1, y_j - 0.5*h_2, EPS);
        }
    }

    double tau = 0.0, r_r, Ar_r, max_norm = 1.0, sum_r_r, sum_Ar_r;
    int iter_cnt = 0, step = 100000, ii = 0;
    double mas[33];
    for (i = 0; i < 33; i++) {
        mas[i] = 0;
    }

    double *right_send_mas, *right_recv_mas, *left_send_mas, *left_recv_mas, *top_send_mas, *top_recv_mas, *down_send_mas, *down_recv_mas;

    // printf("Before memory allocaion, rank %d", rank);

    if(tup_x[0]){
        // выделяем память для второй строки
        top_send_mas = (double *) malloc ((col_end - col_start - 1) * sizeof(double));
        top_recv_mas = (double *) malloc ((col_end - col_start - 1) * sizeof(double));
    }
    if(tup_x[0] + 1 < n){
        // выделяем память для первой строки, если есть вторая
        down_send_mas = (double *) malloc ((col_end - col_start - 1) * sizeof(double));
        down_recv_mas = (double *) malloc ((col_end - col_start - 1) * sizeof(double));
    }
    if(tup_x[1]){
        // выделяем память для первого столбца, если есть второй
        left_send_mas = (double *) malloc ((row_end - row_start - 1) * sizeof(double));
        left_recv_mas = (double *) malloc ((row_end - row_start - 1) * sizeof(double));
    }
    if (tup_x[1] + 1 < k){
        // выделяем память для второго столбца
        right_send_mas = (double *) malloc ((row_end - row_start - 1) * sizeof(double));
        right_recv_mas = (double *) malloc ((row_end - row_start - 1) * sizeof(double));
    }

    // printf("After memory allocaion, rank %d", rank);

    int iter = 0, msg_iter = 0;
    double max_r;

    while (max_norm > delta)
    {
        r_r = 0.0;
        Ar_r = 0.0;
        max_norm = 0.0;
        sum_r_r = 0.0;
        sum_Ar_r = 0.0;
        max_r = -1.0;
        iter_cnt++;

        MPI_Request recv_right_r, recv_left_r, recv_top_r, recv_down_r, recv_right_w, recv_left_w, recv_top_w, recv_down_w;

        int right_send_iter = 0, right_recv_iter = 0, left_send_iter = 0, left_recv_iter = 0, top_send_iter = 0, top_recv_iter = 0, down_send_iter = 0, down_recv_iter = 0;
        if(tup_x[0]){
            // принимаем от первой строки
            MPI_Irecv(top_recv_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]-1)*k + tup_x[1]%n, msg_iter, MPI_COMM_WORLD, &recv_top_r);
        }
        if(tup_x[0] + 1 < n){
            // принимаем от второй строки
            MPI_Irecv(down_recv_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]+1)*k + tup_x[1]%n, msg_iter, MPI_COMM_WORLD, &recv_down_r);
        }
        if(tup_x[1]){
            // принимаем от первого столбца
            MPI_Irecv(left_recv_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1]-1)%n, msg_iter, MPI_COMM_WORLD, &recv_left_r);
        }
        if(tup_x[1] + 1 < k){
            // принимаем от второго столбца
            MPI_Irecv(right_recv_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1] + 1)%n, msg_iter, MPI_COMM_WORLD, &recv_right_r);
        }

        // printf("Before r calculation, rank %d", rank);

        for (i = row_start + 1; i < row_end; i++) {
            for (j = col_start + 1; j < col_end; j++) {
                r[i][j] = -(a[i+1][j] * (w[i+1][j] - w[i][j]) / h_1 - a[i][j] * (w[i][j] - w[i-1][j]) / h_1) / h_1
                            - (b[i][j+1] * (w[i][j+1] - w[i][j]) / h_2 - b[i][j] * (w[i][j] - w[i][j-1]) / h_2) / h_2
                            - F[i][j];
                r_r += r[i][j] * r[i][j];
                if(abs(r[i][j]) > max_r){
                    max_r = abs(r[i][j]);
                }

                if(i == row_start + 1 && tup_x[0]){
                    // для отправки строке 1 при наличии строки 2
                    top_send_mas[top_send_iter++] = r[i][j];
                }
                if(i == row_end - 1 && tup_x[0] + 1 < n){
                    // для отправки строке 2
                    down_send_mas[down_send_iter++] = r[i][j];
                }
                if(j == col_start + 1 && tup_x[1]){
                    // для отправки столбцу 1 при наличии столбца 2
                    left_send_mas[left_send_iter++] = r[i][j];
                }
                if(j == col_end - 1 && tup_x[1] + 1 < k){
                    // для отправки столбцу 2
                    right_send_mas[right_send_iter++] = r[i][j];
                }
                
            }
        }
        r_r *= h_1 * h_2;

        if (iter_cnt % step == 0) {
            mas[ii] = sqrt(r_r);
            ii++;
        }

        // printf("After r calculation, rank %d", rank);

        if(tup_x[0]){
            // отправляем r и получаем w от первой строки при наличии строки 2
            MPI_Request send_up;
            MPI_Isend(top_send_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]-1)*k + tup_x[1]%n, msg_iter, MPI_COMM_WORLD, &send_up);

            MPI_Wait(&recv_top_r, MPI_STATUS_IGNORE);

            for(i = col_start + 1; i < col_end; i++){
                r[row_start][i] = top_recv_mas[top_recv_iter++];
            }

            MPI_Irecv(top_recv_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]-1)*k + tup_x[1]%n, msg_iter+1, MPI_COMM_WORLD, &recv_top_w);
        }
        if(tup_x[0] + 1 < n){
            // отправляем r и получаем w от второй строки
            MPI_Request send_down;
            MPI_Isend(down_send_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]+1)*k + tup_x[1]%n, msg_iter, MPI_COMM_WORLD, &send_down);

            MPI_Wait(&recv_down_r, MPI_STATUS_IGNORE);

            for(i = col_start + 1; i < col_end; i++){
                r[row_end][i] = down_recv_mas[down_recv_iter++];
            }

            MPI_Irecv(down_recv_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]+1)*k + tup_x[1]%n, msg_iter+1, MPI_COMM_WORLD, &recv_down_w);
        }
        if(tup_x[1]){
            // отправляем r и получаем w от первому столбцу при наличии столбца 2
            MPI_Request send_left;
            MPI_Isend(left_send_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1]-1)%n, msg_iter, MPI_COMM_WORLD, &send_left);

            MPI_Wait(&recv_left_r, MPI_STATUS_IGNORE);

            for(i = row_start + 1; i < row_end; i++){
                r[i][col_start] = left_recv_mas[left_recv_iter++];
            }

            MPI_Irecv(left_recv_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1]-1)%n, msg_iter+1, MPI_COMM_WORLD, &recv_left_w);
        }
        if(tup_x[1] + 1 < k){
            // отправляем r и получаем w от второго столбца
            MPI_Request send_right;
            MPI_Isend(right_send_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1] + 1)%n, msg_iter, MPI_COMM_WORLD, &send_right);

            MPI_Wait(&recv_right_r, MPI_STATUS_IGNORE);

            for(i = row_start + 1; i < row_end; i++){
                r[i][col_end] = right_recv_mas[right_recv_iter++];
            }

            MPI_Irecv(right_recv_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1] + 1)%n, msg_iter+1, MPI_COMM_WORLD, &recv_right_w);
        }

        // printf("After r sending, rank %d", rank);

        left_send_iter = 0;
        right_send_iter = 0;
        top_send_iter = 0;
        down_send_iter = 0;
        right_recv_iter = 0;
        left_recv_iter = 0;
        top_recv_iter = 0;
        down_recv_iter = 0;

        msg_iter++;

        for (i = row_start + 1; i < row_end; i++) {
            for (j = col_start + 1; j < col_end; j++) {
                Ar[i][j] = -(a[i+1][j] * (r[i+1][j] - r[i][j]) / h_1 - a[i][j] * (r[i][j] - r[i-1][j]) / h_1) / h_1
                            - (b[i][j+1] * (r[i][j+1] - r[i][j]) / h_2 - b[i][j] * (r[i][j] - r[i][j-1]) / h_2) / h_2;
                Ar_r += Ar[i][j] * r[i][j];
            }
        }
        Ar_r *= h_1 * h_2;

        // printf("After Ar calculation, rank %d", rank);

        MPI_Allreduce(&r_r, &sum_r_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&Ar_r, &sum_Ar_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        tau = sum_r_r / sum_Ar_r;

        // printf("After first all_reduce, rank %d", rank);
        
        for (i = row_start + 1; i < row_end; i++) {
            for (j = col_start + 1; j < col_end; j++) {
                double prev_value = w[i][j];
                w[i][j] = prev_value - tau * r[i][j];

                w_rotated[j][i] = w[i][M-j-1];

                if(i == row_start + 1 && tup_x[0]){
                    // для отправки первой строке при наличии строки 2
                    top_send_mas[top_send_iter++] = w[i][j];
                }
                if(i == row_end - 1 && tup_x[0] + 1 < n){
                    // для отправки второй строке
                    down_send_mas[down_send_iter++] = w[i][j];
                }
                if(j == col_start + 1 && tup_x[1]){
                    // для отправки первому столбцу при наличии столбца 2
                    left_send_mas[left_send_iter++] = w[i][j];
                }
                if(j == col_end - 1 && tup_x[1] + 1 < k){
                    // для отправки второму столбцу
                    right_send_mas[right_send_iter++] = w[i][j];
                }
            }
        }

        // printf("After w calculation, rank %d", rank);

        if(tup_x[0]){
            // отправляем w и получаем w от первой строки при наличии строки 2
            MPI_Request send_up;
            MPI_Isend(top_send_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]-1)*k + tup_x[1]%n, msg_iter, MPI_COMM_WORLD, &send_up);

            MPI_Wait(&recv_top_w, MPI_STATUS_IGNORE);

            for(j = col_start + 1; j < col_end; j++){
                w[row_start][j] = top_recv_mas[top_recv_iter++];
            }

        }
        if(tup_x[0] + 1 < n){
            // отправляем w и получаем w от второй строки
            MPI_Request send_down;
            MPI_Isend(down_send_mas, col_end - col_start - 1, MPI_DOUBLE, (tup_x[0]+1)*k + tup_x[1]%n, msg_iter, MPI_COMM_WORLD, &send_down);

            MPI_Wait(&recv_down_w, MPI_STATUS_IGNORE);

            for(j = col_start + 1; j < col_end; j++){
                w[row_end][j] = down_recv_mas[down_recv_iter++];
            }
        }
        if(tup_x[1]){
            // отправляем w и получаем w от первого столбца при наличии столбца 2
            MPI_Request send_left;
            MPI_Isend(left_send_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1]-1)%n, msg_iter, MPI_COMM_WORLD, &send_left);

            MPI_Wait(&recv_left_w, MPI_STATUS_IGNORE);

            for(i = row_start + 1; i < row_end; i++){
                w[i][col_start] = left_recv_mas[left_recv_iter++];
            }
        }
        if(tup_x[1] + 1 < k){
            // отправляем w и получаем w от второго столбца
            MPI_Request send_right;
            MPI_Isend(right_send_mas, row_end - row_start - 1, MPI_DOUBLE, tup_x[0]*k + (tup_x[1] + 1)%n, msg_iter, MPI_COMM_WORLD, &send_right);

            MPI_Wait(&recv_right_w, MPI_STATUS_IGNORE);

            for(i = row_start + 1; i < row_end; i++){
                w[i][col_end] = right_recv_mas[right_recv_iter++];
            }
        }

        // printf("After w sending, rank %d", rank);
        
        MPI_Allreduce(&max_r, &max_norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // printf("After second all_reduce, rank %d", rank);

        iter++;
        msg_iter++;

    } // while

    if(tup_x[0]){
        free(top_send_mas);
        free(top_recv_mas);
    }
    if(tup_x[0] + 1 < n){
        free(down_send_mas);
        free(down_recv_mas);
    }
    if(tup_x[1]){
        free(left_send_mas);
        free(left_recv_mas);
    }
    if(tup_x[1] + 1 < k){
        free(right_send_mas);
        free(right_recv_mas);
    }

    if(rank == 0) {
        printf("Programm finished for %lf seconds\n", MPI_Wtime()-start_time);
        printf("Iterations count: %d\n", iter);
    }
    
    MPI_Finalize();

    return 0;
}