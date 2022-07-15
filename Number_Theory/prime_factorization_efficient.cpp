// can also return an vector instead of map

map<int, int> factorize(int x) {
    map<int, int> ret;
    for(int i = 2; i * i <= x; i ++) {
        while(x % i == 0) {
            ret[i] ++;
            x /= i;
        }
    }
    if(x > 1) ret[x] ++;
    return ret;
}