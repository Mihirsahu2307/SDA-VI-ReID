const int N = 1e5 + 5, M = 1e9 + 7;
int fact[N], ifact[N],

int binpow(int a, int b) {
    a %= M;
    int res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % M;
        a = a * a % M;
        b >>= 1;
    }
    return res;
}

void fac(int n) {
	fact[0] = 1;
	for(int i = 1; i <= n; i ++) {
		fact[i] = fact[i - 1] * i % M;
	}
}

// computes inv fact for each i from 1 to n
void invfact(int n) {
	ifact[n] = binpow(fact[n], M - 2); // proof: FLT
	for(int i = n - 1; i >= 1; i --) {
		ifact[i] = (i + 1) * ifact[i + 1] % M;
	}
}