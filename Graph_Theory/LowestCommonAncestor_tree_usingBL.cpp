#include <bits/stdc++.h>
using namespace std;

#define fo(a, b, c) for(int(a) = (b); (a) < (c); (a)++)
#define rev(a, b, c) for(int(a) = b; a >= c; a--)
#define pb push_back
#define int long long
#define vi vector<int>
#define pii pair<int, int>
#define F first
#define S second
#define endl "\n"
#define mem(a,b) memset(a,(b),sizeof(a))
#define vpii vector<pair<int, int>>
#define all(v) v.begin(), v.end()
#define showall(a, n) fo(i, 0, n){cout << a[i] << ' ';} cout << endl;
#define show2(a, b) cout << a <<  ' ' << b << endl;
#define fastio ios_base::sync_with_stdio(0);cin.tie(0)
#define YES(b) if(b) cout<<"YES"<<endl; else cout<<"NO"<<endl;
#define CASE(t, a) cout<<"Case #"<<t<<": "<<a<<endl;
const int M = 1000000007;


const int N = 1e4 + 5, lg = 20;
vi G[N];
int lvl[N] = {}; // lg > log(N)
int up[N][lg];


// building the up table:
void dfs(int u, int p) {
	up[u][0] = p;
	fo(j, 1, lg) {
		up[u][j] = up[up[u][j - 1]][j - 1];
	}

	for(auto c : G[u]) {
		if(c == p) continue;
		lvl[c] = lvl[u] + 1;
		dfs(c, u);
	}
}

// query(u, v) returns LCA
// Binary lifting works like binary search, here we move up or down depending on condition
int query(int u, int v) {
	if(lvl[u] < lvl[v]) {
		swap(u, v);
	}

	// bring u to the level of v
	int k = lvl[u] - lvl[v];
	rev(j, lg - 1, 0) {
		if(k & (1ll << j)) {
			u = up[u][j];
		}
	}
	// don't forget this case:
	if(u == v) {
		return u;
	}

	// maintain this reversed order here (just iterate in reverse for BL in general)
	rev(j, lg - 1, 0) {
		if(up[u][j] != up[v][j]) {
			u = up[u][j];
			v = up[v][j];
		}
	}
	return up[u][0];
}

signed main() {
    fastio;

    /*
     COMMENT OUT BEOFRE SUBMISSION
     */
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif

    int T = 1;
    // cin >> T;
    fo(t, 1, T + 1) {
    	int n, a, b, m;
    	cin >> n;
    	fo(i, 0, n) {
    		cin >> m;
    		while(m --) {
    			cin >> a;
    			G[i].pb(a);
    			G[a].pb(i);
    		}
    	}
    	dfs(0, 0);
    	cin >> m;
    	// cout << m << endl;
    	while(m --) {
    		cin >> a >> b;
    		cout << query(a, b) << endl;
    	}
    }

    return 0; 
}