// can still be used to find prefix sums in the same way
// eg. query(10) returns sum upto element = 10, so no need to change main function at all

// 1-indexing
const int N = 1e6 + 6, lg = 28;
int tree[N] = {};

void update(int in, int val) {
    for (int i = in; i < N; i += i & -i) {
        tree[i] += val;
    }
}

int query(int r) {
    int res = 0;
    for (int i = r; i > 0; i -= i & -i) {
        res += tree[i];
    }
    return res;
}


signed main() {
    ios_base::sync_with_stdio(0);cin.tie(0);
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif

    int T = 1;
    cin >> T;
    for(int t = 1; t < T + 1; t ++) {
        
    }

    return 0; 
}