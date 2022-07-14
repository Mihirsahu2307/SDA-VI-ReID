#include <bits/stdc++.h>
#define int long long
#define endl "\n"

using namespace std;

// set k = d - c
// need to count (i, j) i < j, st. a[i] <= a[j] + k

// idea:
// during coordinate compression, use both a[i] and a[i] + k values
// create 2 arrays X[i] = a[i] and Y[i] = a[i] + k and transform values after compression
// iterate over i in Y and update tree as per values in X
// this way we are counting (i, j), i < j, X[i] <= Y[j]

// note that here, size of fenwick tree must be 2*n atleast

const int N = 2e5 + 5;
int tree[N] = {};

int bruteForce(int a[], int b[], int c, int d, int n) {
    int ans = 0;
    for(int i = 0; i < n; i ++) {
        for(int j = i + 1; j < n; j ++) {
            if(a[i] - a[j] + c <= b[i] - b[j] + d) {
                ans ++;
            } 
        }
    }

    return ans;
}

void update(int index, int inc) {
    for(int i = index; i < N; i = (i | (i + 1))) {
        tree[i] += inc;
    }
}

int query(int r) {
    int sum = 0;
    for(int i = r; i >= 0; i = ((i & (i + 1)) - 1)) {
        sum += tree[i];
    }
    return sum;
}

signed main() {
    ios_base::sync_with_stdio(0); cin.tie(0);
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    int T = 1;
    cin >> T;
    for (int t = 1; t < T + 1; t ++) {
        memset(tree, 0, sizeof tree);
        int n, c, d;
        cin >> n >> c >> d;
        int a[n], b[n];
        for(int i = 0 ; i < n; i ++) {
            cin >> a[i];
        }
        for(int i = 0 ; i < n; i ++) {
            cin >> b[i];
        }

        cout << bruteForce(a, b, c, d, n) << endl;

        // find (i, j) i < j, st. a[i] - a[j] + c <= b[i] - b[j] + d

        set<int> all;
        map<int, int> compress;
        int x[n], y[n];
        for(int i = 0; i < n; i ++) {
            x[i] = a[i] - b[i];
            y[i] = x[i] + d - c;

            all.insert(x[i]);
            all.insert(y[i]);
        }

        int coord = 0;
        for(auto e : all) {
            compress[e] = coord ++;
        }

        for(int i = 0 ; i < n; i ++) {
            x[i] = compress[x[i]];
            y[i] = compress[y[i]];
        }

        int ans = 0;
        update(x[0], 1);
        for(int i = 1; i < n; i ++) {
            ans += query(y[i]);
            update(x[i], 1);
        }

        cout << ans << endl;
    }

    return 0;
}
