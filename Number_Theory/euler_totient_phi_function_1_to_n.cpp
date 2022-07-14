const int N = 1e6 + 5;
int Phi[N] = {};

void PHI(int n) {
    // initialize every phi(n) with n and then perform iterative multiplication
    // if Phi[i] == 1, it means i is a prime number, just like sieve idea

    for(int i = 0; i <= n; i ++) {
        Phi[i] = i;
    }

    for(int i = 2; i <= n; i ++) {
        if(Phi[i] == i) { // without this if statement, complexity becomes n logn
            for(int j = i; j <= n; j += i) {
                Phi[j] -= (Phi[j] / i);
            }
        }
    }
}