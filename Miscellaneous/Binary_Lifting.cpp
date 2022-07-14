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

// Problem link: https://www.codechef.com/problems/SPECIALSTR

// note that usually the up table has to be built during dfs
// so that the ancestors of current ancestor are already known

// for finding k_th ancestor, just iterate over all bits and for each set bit j in k
// set node = up[node][j];

const int N = 1e6 + 5, lg = 21;
int up[N][lg];

// Important**
int query(int in, int k) {
    int j = 0, ret = in;
    while(k) {
        if(k % 2) {
            ret = up[ret][j];
            if(ret == M) break;  // **Remember**
        }
        j ++;
        k /= 2;
    }

    return ret;
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
    cin >> T;
    string alpha = "abcdefghijklmnopqrstuvwxyz";
    fo(t, 1, T + 1) {
        int n, m;
        cin >> n;
        fo(i, 0, n) {
            fo(j, 0, lg) {
                up[i][j] = M;
            }
        }
        string s;
        cin >> s >> m;
        set<int> ind[26];
        fo(i, 0, n) {
            ind[s[i] - 'a'].insert(i);
        }

        // Important**
        rev(i, n - 1, 0) {
            int next = (s[i] - 'a' + 1) % 26;
            auto it = ind[next].upper_bound(i);
            if(it == ind[next].end()) {
                up[i][0] = M;
            }
            else {
                up[i][0] = *it;
            }

            // Always initialize up[i][0] separately as per need
            fo(j, 1, lg) {
                if(up[i][j - 1] == M) up[i][j] = M;  // **Remember**
                else up[i][j] = up[up[i][j - 1]][j - 1];
            }
        }

        int ans = M;
        fo(i, 0, n) {
            if(s[i] == 'a') {
                ans = min(ans, query(i, m - 1) - i - m + 1);
            }
        }
        if(ans >= n) {
            cout << -1 << endl;
        }
        else {
            cout << ans << endl;
        }
    }

    return 0; 
}
