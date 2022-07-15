void linearPrefixMex(int a[], int n, int r) {
	// r is the max element of a
	// works only when range of a is small
	// if range of a is large, then stick to n logn solution using set of 
	// unvisited elements and pick the smallest unvisited element as the mex

	// observe that prefix mex will either increase or remain constant
	// and it can only increase upto r + 1
	
	int pref[n];
	int mex = 0, vis[r + 1] = {};
	fo(i, 0, n) {
		vis[a[i]] ++;
		while(vis[mex]) mex ++; 
		pref[i] = mex;
	}
}


// O(n + r)
// represent: increment in pref as * and incr in vis arr as a |
// n stars and r bars
// **** .. (n) ||||.. (r) ==> total = n + r in any permutation