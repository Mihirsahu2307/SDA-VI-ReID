// sieve to calculate lowest prime factor for all numbers upto N
// Can be used for logN factorization

// while x > 1: x := x / lp[x] ==> works in log N as lp[x] >= 2

int lp[N] = {};
void sieve() {
    for(int i = 2; i < N; i ++) {
        lpf[i] = i;
    }
    // Caveat: Don't forget i*i < N or else j = i*i will overflow integer type ==> RTE
    // Or you may use int ~ long long (not preferred for NT problems)
    for(int i = 2; i * i < N; i ++) {
        if(lpf[i] == i) {
            for(int j = i * i; j < N; j += i) {
                // Always use this if statement: Significantly improves run time
                // Don't do lpf[j] = min(lpf[j], i)
                if(lpf[j] == j) {
                    lpf[j] = i;
                }
            }
        }
    }
}