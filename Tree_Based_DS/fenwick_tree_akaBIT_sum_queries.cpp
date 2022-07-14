#include <bits/stdc++.h>
#define int long long
#define endl "\n"
#define fo(i, a, b) for(int i = a; i < b; i ++)

using namespace std;


// note space requirement by fenwick tree is n + 1 when the size of input array is n

// 0-indexing
const int N = 1e5 + 5;
int fen[N];
void update(int index, int inc) {
    for(int i = index; i < N; i = i | (i + 1)) {
        fen[i] += inc; // make sure to use difference (inc) not the new value at a[i]
    }
}

int query(int r) {
    int res = 0;
    for(int i = r; i >= 0; i = (i & (i + 1)) - 1) {
        res += fen[i];
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