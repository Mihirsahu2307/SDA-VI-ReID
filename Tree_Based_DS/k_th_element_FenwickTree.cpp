#include <bits/stdc++.h>
// #define int long long
#define endl "\n"

using namespace std;

// Problem: https://codeforces.com/contest/1354/problem/D
// Note: For using find_k_th method, prefer using 1-indexing, in general try using
// 1-indexing when 0-indexing gives WA

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


int find_k_th(int k) {
    int sum = 0, in = 0;
    for (int i = lg; i >= 0; i --) {
        if (in + (1 << i) < N && sum + tree[in + (1 << i)] < k) {
            sum += tree[in  + (1 << i)];
            in += (1 << i);
        }
    }

    return in + 1;
}

signed main() {
    ios_base::sync_with_stdio(0); cin.tie(0);
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    int T = 1;
    // cin >> T;
    for (int t = 1; t < T + 1; t ++) {
        int n, q, a;
        cin >> n >> q;

        for (int i = 0; i < n; i ++) {
            cin >> a;
            update(a, 1);
        }

        // that means 7 translates to 8

        for (int i = 0 ; i < q; i ++) {
            cin >> a;
            if (a > 0) {
                update(a, 1);
            }
            else {
                int ele = find_k_th(-a);
                // cout << ele << endl;
                update(ele, -1);
            }
        }

        // cout << find_k_th(0) << endl;

        if (find_k_th(1) > 1e6) {
            cout << 0 << endl;
        }
        else {
            cout << find_k_th(1) << endl;
        }
    }

    return 0;
}