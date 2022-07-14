const int N = 1e7;
int prime[N];

void sieve(int n) {
    memset(prime, 1, sizeof prime);
    prime[0] = prime[1] = 0;
    for(int i = 2; i * i <= n; i ++) {
        if(prime[i]) {
            for(int j = i * i; j <= n; j += i) {
                prime[j] = 0;
            }
        }
    }
}
